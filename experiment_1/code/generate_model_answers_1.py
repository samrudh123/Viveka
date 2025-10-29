import argparse
import json
import os
import gc  # Import the garbage collection module

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import set_seed

# Assume these utils are in a separate file or defined elsewhere in your project
from compute_correctness import compute_correctness
from probing_utils import (LIST_OF_DATASETS, LIST_OF_MODELS,
                           MODEL_FRIENDLY_NAMES, generate,
                           load_model_and_validate_gpu, tokenize)


def parse_args():
    parser = argparse.ArgumentParser(description="A script for generating model answers and outputting to csv")
    parser.add_argument("--model",
                        choices=LIST_OF_MODELS,
                        required=True)
    parser.add_argument("--dataset",
                        choices=LIST_OF_DATASETS)
    parser.add_argument("--verbose", action='store_true', help='print more information')
    parser.add_argument("--n_samples", type=int, help='number of examples to use', default=None)
    parser.add_argument("--train_size", type=int,help='train size of datasets',default=2000)
    # NEW: Add a command-line argument for batch size for flexibility
    parser.add_argument("--batch_size", type=int, help='Number of prompts to process at a time to save RAM', default=10)

    return parser.parse_args()


# ==============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS (Unchanged)
# ==============================================================================
def load_data_movies(test=False):
    file_name = 'movie_qa'
    if test:
        file_path = f'../data/{file_name}_test.csv'
    else: # train
        file_path = f'../data/{file_name}_train.csv'
    if not os.path.exists(file_path):
        data = pd.read_csv(f"../data/{file_name}.csv")
        train, test = train_test_split(data, train_size=10000, random_state=42)
        train.to_csv(f"../data/{file_name}_train.csv", index=False)
        test.to_csv(f"../data/{file_name}_test.csv", index=False)
    data = pd.read_csv(file_path)
    return data['Question'], data['Answer']

def load_data_nli(split, data_file_names):
    data_folder = '../data'
    file_path = f"{data_folder}/{data_file_names[split]}.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = pd.read_csv(file_path)
    return data['Question'], data['Answer'], data['Origin']

def load_data_snli(split):
    return load_data_nli(split, {'train': 'snli_train', 'test': 'snli_validation'})

def load_data_mnli(split):
    return load_data_nli(split, {'train': 'mnli_train', 'test': 'mnli_validation'})

def load_data_nq(split, with_context=False):
    file_name = 'nq_wc'
    file_path = f'../data/{file_name}_dataset_{split}.csv'
    if not os.path.exists(file_path):
        all_data = pd.read_csv(f"../data/{file_name}_dataset.csv")
        train, test = train_test_split(all_data, train_size=10000, random_state=42)
        train.to_csv(f"../data/{file_name}_dataset_train.csv", index=False)
        test.to_csv(f"../data/{file_name}_dataset_test.csv", index=False)
    data = pd.read_csv(file_path)
    context_data = data['Context'] if with_context else None
    return data['Question'], data['Answer'], context_data

def load_data_winogrande(split):
    file_path = f"../data/winogrande_{split}.csv"
    if not os.path.exists(file_path):
        all_data = pd.read_csv(f"../data/winogrande.csv")
        train, test = train_test_split(all_data, train_size=10000, random_state=42)
        train.to_csv(f"../data/winogrande_train.csv", index=False)
        test.to_csv(f"../data/winogrande_test.csv", index=False)
    data = pd.read_csv(file_path)
    return data['Question'], data['Answer'], data['Wrong_Answer']

def load_data_triviaqa(test=False, legacy=False, train_size=2000):
    if test:
        file_path = '../data/triviaqa-unfiltered/unfiltered-web-dev.json'
    else:
        file_path = '../data/triviaqa-unfiltered/unfiltered-web-train.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['Data']
    data, _ = train_test_split(data, train_size=train_size, random_state=42)
    return [ex['Question'] for ex in data], [ex['Answer']['Aliases'] for ex in data]

def load_data_math(test=False):
    file_name = "AnswerableMath_test.csv" if test else "AnswerableMath.csv"
    data = pd.read_csv(f"../data/{file_name}")
    answers = data['answer'].map(lambda x: eval(x)[0])
    return data['question'], answers

def load_data_imdb(split):
    dataset = load_dataset("imdb")
    indices = np.arange(len(dataset[split]))
    np.random.shuffle(indices)
    subset = dataset[split].select(indices[:10000])
    return subset['text'], subset['label']

def load_winobias(dev_or_test):
    data = pd.read_csv(f'../data/winobias_{dev_or_test}.csv')
    return (data['sentence'], data['q'], data['q_instruct']), data['answer'], data['incorrect_answer'], data['stereotype'], data['type']

def load_hotpotqa(split, with_context):
    dataset = load_dataset("hotpot_qa", 'distractor')
    np.random.seed(42)
    subset_indices = np.random.choice(len(dataset[split]), 10000, replace=False)
    
    questions = [dataset[split][int(x)]['question'] for x in subset_indices]
    labels = [dataset[split][int(x)]['answer'] for x in subset_indices]

    if with_context:
        prompts = []
        for idx in subset_indices:
            context_text = ""
            for sentences in dataset[split][int(idx)]['context']['sentences']:
                context_text += " ".join(sentences) + "\n"
            prompts.append(context_text.strip() + "\n" + dataset[split][int(idx)]['question'])
        questions = prompts
        
    return questions, labels

def load_data(dataset_name, train_size):
    max_new_tokens = 100
    context, origin, stereotype, type_, wrong_labels = None, None, None, None, None
    
    if dataset_name == 'triviaqa':
        all_questions, labels = load_data_triviaqa(False, train_size=train_size)
        preprocess_fn = triviqa_preprocess
    # ... Add other datasets as in your original script
    else:
        raise TypeError(f"Data type '{dataset_name}' is not supported in this example")
        
    return all_questions, context, labels, max_new_tokens, origin, preprocess_fn, stereotype, type_, wrong_labels

def triviqa_preprocess(model_name, all_questions, labels):
    if 'instruct' in model_name.lower():
        return all_questions
    return [f'Q: {q}\nA:' for q in all_questions]

def nq_preprocess(model_name, all_questions, labels, with_context, context):
    # Simplified for brevity
    return triviqa_preprocess(model_name, all_questions, labels)

# ==============================================================================
# CORE GENERATION AND ACTIVATION EXTRACTION LOGIC (Unchanged)
# ==============================================================================

def generate_model_answers(data, model, tokenizer, device, model_name, do_sample=False,
                           temperature=1.0, top_p=1.0, max_new_tokens=100, stop_token_id=None, verbose=False):
    all_textual_answers = []
    all_input_output_ids = []
    
    for prompt in data: # Removed tqdm here to have a per-batch progress bar
        model_input = tokenize(prompt, tokenizer, model_name).to(device)
        with torch.no_grad():
            model_output = generate(model_input, model, model_name, do_sample,
                                    output_scores=False,
                                    max_new_tokens=max_new_tokens,
                                    top_p=top_p, temperature=temperature,
                                    stop_token_id=stop_token_id, tokenizer=tokenizer)
        answer = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):], skip_special_tokens=True)
        all_textual_answers.append(answer)
        all_input_output_ids.append(model_output['sequences'][0].cpu())

    return all_textual_answers, all_input_output_ids


def get_final_residual_stream(model, all_input_output_ids, device):
    all_activations = []
    model.eval()

    for full_ids in all_input_output_ids: # Removed tqdm here
        input_ids = full_ids.to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        
        residual_stream = torch.stack([
            s.squeeze(0).cpu().to(torch.float16) for s in outputs.hidden_states[1:]
        ])
        all_activations.append(residual_stream)
        
    return all_activations


# ==============================================================================
# MAIN EXECUTION SCRIPT (Modified for RAM efficiency)
# ==============================================================================

def init_wandb(args):
    cfg = vars(args)
    wandb.init(
        project="generate_answers_with_activations_ram_efficient",
        config=cfg
    )

def main():
    args = parse_args()
    init_wandb(args)
    set_seed(42)
    dataset_size = args.n_samples
    BATCH_SIZE = args.batch_size # Use batch size from args

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_validate_gpu(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stop_token_id = None
    if 'instruct' not in args.model.lower():
        stop_token_id = tokenizer.encode('\n', add_special_tokens=False)[-1]

    print(f"Loading dataset: {args.dataset}...")
    all_questions, context, labels, max_new_tokens, _, preprocess_fn, _, _, wrong_labels = load_data(args.dataset, args.train_size)

    # Create a directory for batched activations
    model_name_safe = MODEL_FRIENDLY_NAMES[args.model]
    output_dir = f"../output/{model_name_safe}_{args.dataset}_activations_batched"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path_answers_csv = f"../output/{model_name_safe}-answers-{args.dataset}.csv"

    if dataset_size:
        all_questions = all_questions[:dataset_size]
        labels = labels[:dataset_size]
        # (add other list slicing as needed)
    
    print("Preprocessing all prompts...")
    prompts = preprocess_fn(args.model, all_questions, labels)
    
    # This list will collect the small CSV data from each batch
    all_results_for_csv = []
    
    print(f"Starting processing in batches of {BATCH_SIZE}...")
    num_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(num_batches), desc="Overall Progress"):
        # --- 1. Get the current batch ---
        start_index = i * BATCH_SIZE
        end_index = start_index + BATCH_SIZE
        batch_prompts = prompts[start_index:end_index]
        batch_raw_questions = all_questions[start_index:end_index]
        batch_labels = labels[start_index:end_index]
        batch_wrong_labels = wrong_labels[start_index:end_index] if wrong_labels else None

        # --- 2. Generate answers and get token IDs for the batch ---
        batch_model_answers, batch_input_output_ids = generate_model_answers(
            batch_prompts, model, tokenizer, device, args.model,
            max_new_tokens=max_new_tokens, stop_token_id=stop_token_id
        )

        # --- 3. Extract activations for the batch ---
        batch_activations = get_final_residual_stream(model, batch_input_output_ids, device)

        # --- 4. Save this batch's activations immediately to disk ---
        batch_activations_path = os.path.join(output_dir, f"activations_batch_{i}.pt")
        torch.save(batch_activations, batch_activations_path)

        # --- 5. Free up memory ---
        del batch_activations
        del batch_input_output_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 6. Compute correctness and collect CSV data for the batch ---
        res = compute_correctness(batch_prompts, args.dataset, args.model, batch_labels, model, batch_model_answers, tokenizer, batch_wrong_labels)
        
        for j in range(len(batch_prompts)):
            all_results_for_csv.append({
                'raw_question': batch_raw_questions[j],
                'question_prompt': batch_prompts[j],
                'model_answer': batch_model_answers[j],
                'correct_answer': batch_labels[j],
                'automatic_correctness': res['correctness'][j]
            })

    # --- After all batches are processed ---
    
    # Create and save the final CSV file
    print(f"Saving answers and metadata to {file_path_answers_csv}...")
    final_df = pd.DataFrame(all_results_for_csv)
    final_df.to_csv(file_path_answers_csv, index=False)

    # Log final accuracy to wandb
    final_accuracy = final_df['automatic_correctness'].mean()
    wandb.summary[f'final_accuracy'] = final_accuracy
    print(f"Final Accuracy: {final_accuracy:.4f}")
    
    print("Script finished successfully!")
    print(f"Batched activations saved in: {output_dir}")

if __name__ == "__main__":
    main()