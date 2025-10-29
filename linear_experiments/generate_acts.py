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
            a = attn_hooks[l].out[0, -1].detach()  #this is the place where we take only the -1 token for probing
                                                   #what we need here is the exact answer token. 
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
            'cities', 'sp_en_trans', 'inventors', 'animal_class', 'element_symb', 'facts',
            'neg_cities', 'neg_sp_en_trans', 'neg_inventors', 'neg_animal_class', 'neg_element_symb', 'neg_facts',
            'cities_conj', 'sp_en_trans_conj', 'inventors_conj', 'animal_class_conj', 'element_symb_conj', 'facts_conj',
            'cities_disj', 'sp_en_trans_disj', 'inventors_disj', 'animal_class_disj', 'element_symb_disj', 'facts_disj',
            'larger_than', 'smaller_than', 'cities_de', 'neg_cities_de', 'sp_en_trans_de', 'neg_sp_en_trans_de',
            'inventors_de', 'neg_inventors_de', 'animal_class_de', 'neg_animal_class_de', 'element_symb_de',
            'neg_element_symb_de', 'facts_de', 'neg_facts_de'
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
