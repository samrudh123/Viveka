import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser
import glob

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        # module_outputs is a tuple (attn_out, attn_weights) or single tensor for MLP
        if isinstance(module_outputs, tuple):
            self.out = module_outputs[0]
        else:
            self.out = module_outputs


def load_model(model_family: str, model_size: str, model_type: str, device: str):
    """
    Load tokenizer and model, cast precision, move to device, and expose layer list
    """
    model_path = os.path.join(
        config[model_family]['weights_directory'],
        config[model_family][f'{model_size}_{model_type}_subdir']
    )
    # Instantiate tokenizer and model
    if model_family == 'Llama2':
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path)
        tokenizer.bos_token = '<s>'
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

    # Precision casting
    if model_family.lower() == 'gemma2' and t.cuda.is_available() and t.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=t.bfloat16)
    else:
        model = model.to(device=device, dtype=t.float16 if device.startswith('cuda') else t.float32)

    # Expose a consistent layer container
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        layers = model.transformer.layers
    else:
        raise ValueError("Unable to locate transformer layers in model")

    return tokenizer, model, layers


def load_statements(dataset_name):
    """
    Load statements from CSV file and return as list of strings
    """
    path = f"datasets/{dataset_name}.csv"
    df = pd.read_csv(path)
    return df['statement'].tolist()


def get_acts(statements, tokenizer, model, layers, layer_indices, device):
    """
    Attach hooks to both self-attention and MLP blocks at specified layers,
    run statements, and return dict of activations keyed by pseudo-layer.
    """
    attn_hooks, mlp_hooks = {}, {}
    handles = []
    # Register hooks
    for l in layer_indices:
        hook_a = Hook()
        hook_m = Hook()
        layer_mod = layers[l]
        handles += [
            layer_mod.self_attn.register_forward_hook(hook_a),
            layer_mod.mlp.register_forward_hook(hook_m)
        ]
        attn_hooks[l] = hook_a
        mlp_hooks[l] = hook_m

    # Prepare storage: pseudo-layer 2*l for attn, 2*l+1 for mlp
    acts = {2*l: [] for l in layer_indices}
    acts.update({2*l+1: [] for l in layer_indices})

    # Forward through model for each statement
    for stmt in tqdm(statements, desc="Encoding statements"):
        ids = tokenizer.encode(stmt, return_tensors='pt', add_special_tokens=True).to(device)
        model(ids)
        for l in layer_indices:
            a = attn_hooks[l].out[0, -1].detach()
            m = mlp_hooks[l].out[0, -1].detach()
            acts[2*l].append(a)
            acts[2*l+1].append(m)

    # Stack into tensors
    for k in list(acts.keys()):
        acts[k] = t.stack(acts[k]).cpu().float()

    # Remove hooks
    for h in handles:
        h.remove()

    return acts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract attention and MLP activations for statements"
    )
    parser.add_argument('--model_family', default='Llama3',
                        help='Model family: Llama2, Llama3, Gemma, Gemma2, Mistral')
    parser.add_argument('--model_size', default='8B', help='Model size (e.g. 8B, 70B)')
    parser.add_argument('--model_type', default='base', help='base or chat')
    parser.add_argument('--layers', nargs='+', required=True,
                        help='Layer indices to extract from (or -1 for all)')
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Dataset names (without .csv)')
    parser.add_argument('--output_dir', default='acts', help='Root directory for saving activations')
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    args = parser.parse_args()

    # Determine datasets
    ds = args.datasets
    if ds == ['all_topic_specific']:
        ds = [
          "cities_If_then","neg_cities_If_then","inventors_If_then","neg_inventors_If_then","sp_en_trans_If_then","neg_sp_en_trans_If_then","animal_class_If_then","neg_animal_class_If_then",
          "element_symb_If_then","neg_element_symb_If_then"
        ]
    elif ds == ['all']:
        ds = [os.path.relpath(fp, 'datasets').replace('.csv','')
              for fp in glob.glob('datasets/**/*.csv', recursive=True)]

    li = [int(x) for x in args.layers]

    t.set_grad_enabled(False)
    tokenizer, model, layer_modules = load_model(
        args.model_family, args.model_size, args.model_type, args.device
    )
    total_layers = len(layer_modules)
    if li == [-1]:
        li = list(range(total_layers))

    for dataset in ds:
        stmts = load_statements(dataset)
        save_base = os.path.join(
            args.output_dir, args.model_family, args.model_size, args.model_type, dataset
        )
        os.makedirs(save_base, exist_ok=True)
        batch_size = 25
        for start in range(0, len(stmts), batch_size):
            batch = stmts[start:start+batch_size]
            acts = get_acts(batch, tokenizer, model, layer_modules, li, args.device)
            for pseudo, tensor in acts.items():
                filename = os.path.join(
                    save_base, f"layer_{pseudo}_{start}.pt"
                )
                t.save(tensor, filename)
