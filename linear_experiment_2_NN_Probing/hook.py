from utils import (create_prompts, generate_model_answers)
import json
import os
from tqdm import tqdm
import torch as t
from thefuzz import fuzz

import torch
import gc

from transformer_lens import HookedTransformer

class Hook:
    """Hook class compatible with both PyTorch forward hooks and TransformerLens HookPoints."""
    def __init__(self):
        self.out = None

    def __call__(self, *args, **kwargs):
        # TransformerLens signature: (value, hook)
        if len(args) == 2:
            value, hook = args
            self.out = value.detach().cpu()
            return value

        # PyTorch forward-hook signature: (module, module_inputs, module_outputs)
        elif len(args) == 3:
            module, module_inputs, module_outputs = args
            out = module_outputs[0] if isinstance(module_outputs, tuple) else module_outputs
            self.out = out.detach().cpu()
            return

        else:
            raise ValueError("Hook called with unexpected signature. Expected (value, hook) or (module, inputs, outputs).")


def generate_and_label_answers(
    statements,
    correct_answers,
    tokenizer,
    model,
    device,
    num_generations=32,
    output_dir="current_run",
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=64
):
    """ 
    STAGE 1: Generate answers for a slice of statements in batch,
    label them, and save/update the results in a central JSON cache.
    """
    model_name = model.name_or_path.replace("/", "_") if hasattr(model, 'name_or_path') else 'unknown'
    generations_dir = os.path.join(output_dir, "generations")
    os.makedirs(generations_dir, exist_ok=True)
    generations_cache_path = os.path.join(generations_dir, f"{model_name}_generations.json")

    if os.path.exists(generations_cache_path):
        with open(generations_cache_path, 'r', encoding='utf-8') as f:
            generations_cache = json.load(f)
    else:
        generations_cache = {}

    # Filters out already-generated statements
    batch_statements = []
    batch_correct_answers = []
    for stmt, correct_ans in zip(statements, correct_answers):
        if stmt not in generations_cache:
            batch_statements.append(stmt)
            batch_correct_answers.append(correct_ans)

    if not batch_statements:
        return

    prompts = create_prompts(batch_statements, model_name)

    print(f"Generating using temperature {temperature} and top_p = {top_p}")
    all_generated, _ = generate_model_answers(
        prompts, model, tokenizer, device, model_name,
        max_tokens=max_new_tokens,
        num_return_sequences=num_generations,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

    for i, stmt in enumerate(batch_statements):
        stmt_generations = all_generated[i * num_generations:(i + 1) * num_generations]
        generated_texts = [g.strip() for g in stmt_generations]

        try:
            correct_answers_list = eval(batch_correct_answers[i])
            if not isinstance(correct_answers_list, list):
                correct_answers_list = [str(correct_answers_list)]
        except (SyntaxError, NameError):
            correct_answers_list = [str(batch_correct_answers[i])]

        ground_truth_labels = []
        for text in generated_texts:
            is_match = any(fuzz.partial_ratio(str(ans).lower(), text.lower()) > 90
                           for ans in correct_answers_list)
            ground_truth_labels.append(1 if is_match else 0)

        generations_cache[stmt] = {
            "prompt": prompts[i],
            "generated_answers": generated_texts,
            "ground_truth_labels": ground_truth_labels
        }

    with open(generations_cache_path, 'w', encoding='utf-8') as f:
        json.dump(generations_cache, f, indent=2, ensure_ascii=False)

    print(f"\nGeneration and labeling complete for this slice. Cache updated at '{generations_cache_path}'.")


def get_truth_probe_activations(
    statements,
    tokenizer,
    model,
    model_name_arg,
    layers,
    layer_indices,
    device,
    batch_list,
    batch_size_arg=32,
    output_dir="current_run",
    start_index=0,
    end_index=0
):
    """
    STAGE 2: Load generated answers from the cache for a slice of statements,
    and save the captured activations using the correct global index.
    """
    model_name = model_name_arg
    generations_dir = os.path.join(output_dir, "generations")
    activations_dir = os.path.join(output_dir, "activations", model_name)
    os.makedirs(activations_dir, exist_ok=True)
    generations_cache_path = os.path.join(generations_dir, "generated_completions_20k.json")

    if not os.path.exists(generations_cache_path):
        raise FileNotFoundError(f"Generations cache not found at {generations_cache_path}. Please run the 'generate' stage first.")
    with open(generations_cache_path, 'r', encoding='utf-8') as f:
        generations_cache = json.load(f)

    if -1 in layer_indices:
        layer_indices = list(range(len(layers)))

    for local_idx, stmt in enumerate(tqdm(
        statements,
        desc="Stage : Extracting Activations",
        total=len(statements)
    )):
        global_stmt_idx = start_index + local_idx

        if stmt not in generations_cache:
            print(f"Warning: Statement (Index {global_stmt_idx}) '{stmt[:50]}...' not found in cache. Skipping.")
            continue

        output_exists = all(os.path.exists(os.path.join(activations_dir, f"layer_{l_idx}_stmt_{global_stmt_idx}.pt")) for l_idx in layer_indices)
        if output_exists:
            continue

        generation_data = generations_cache[stmt]
        appended_prompts = []
        final_labels = []
        base_prompt = generation_data["prompt"]
        for answer_text, ground_truth in zip(generation_data["generated_answers"], generation_data["ground_truth_labels"]):
            prompt_true = f"{base_prompt} {answer_text} True"
            prompt_false = f"{base_prompt} {answer_text} False"
            appended_prompts.extend([prompt_true, prompt_false])
            final_labels.extend([ground_truth, 1 - ground_truth])

        batch_size = batch_size_arg # how many answers of the same statement you wanna process at once 
        all_last_token_resid = [[] for _ in range(model.cfg.n_layers)]  # list per layer

        for i in range(0, len(appended_prompts), batch_size):
            batch = appended_prompts[i:i+batch_size]

            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            input_ids = tokens["input_ids"].to(device)

            with torch.no_grad():
                _, cache = model.run_with_cache(input_ids)
                torch.cuda.empty_cache()
            # Extract last-token activations for each layer and move to CPU immediately
            for l_idx in range(model.cfg.n_layers):
                layer_last_token = cache[f"blocks.{l_idx}.hook_resid_post"][:, -1, :].cpu()
                torch.cuda.empty_cache()
                all_last_token_resid[l_idx].append(layer_last_token)
            del cache
            del tokens, layer_last_token
            torch.cuda.empty_cache()

        last_token_resid = [torch.cat(all_last_token_resid[l_idx], dim=0) for l_idx in range(model.cfg.n_layers)]
       
        batch_slice = batch_list[global_stmt_idx]
        offset = 0
        
        num_rows = batch_slice * 2
        stmt_labels = final_labels[offset:offset+num_rows]
        for l_idx in range(model.cfg.n_layers):
            q_acts = last_token_resid[l_idx][offset:offset+num_rows, :]

            data = {
            "activations": q_acts.cpu(),
            "labels": torch.tensor(stmt_labels)
            }
            save_path = os.path.join(
            activations_dir,
            f"layer_{l_idx}_stmt_{global_stmt_idx}.pt"
                )
            torch.save(data, save_path)
        

    t.cuda.empty_cache()
    print(f"\nActivation extraction complete for this slice. Activations saved in '{activations_dir}'.")