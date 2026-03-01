from transformers import AutoModelForCausalLM, AutoTokenizer
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import csv
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
import torch
# import random
# import string

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
INSTRUCT = "Answer the question concisely. Q: {} A:"


MODEL_NAME = {
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2_13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama-3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-v0.2": "mistralai/Mistral-7B-Instruct-v0.2", 
    "mistral-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-2": "google/gemma-2-2b",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "vicuna-v1.5": "lmsys/vicuna-7b-v1.5",
}


def get_model_name(model_name):
    return MODEL_NAME[model_name]

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def load_model_and_tokenizer(model_name, device, torch_dtype=torch.float16):
    """prepare LLM and tokenizer"""
    model_name = get_model_name(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    # Set chat template for models that don't have one
    if "gemma" in model_name.lower() and not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}"
        tokenizer.chat_template += "{% if message['role'] == 'user' %}"
        tokenizer.chat_template += "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
        tokenizer.chat_template += "{% elif message['role'] == 'assistant' %}"
        tokenizer.chat_template += "<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
        tokenizer.chat_template += "{% endif %}"
        tokenizer.chat_template += "{% endfor %}"
        tokenizer.chat_template += "{% if add_generation_prompt %}"
        tokenizer.chat_template += "<start_of_turn>model\n"
        tokenizer.chat_template += "{% endif %}"
    
    return model, tokenizer

def load_bleurt(device):
    """BLEURT model and tokenizer"""
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').to(device)
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
    model.eval()
    
    return model, tokenizer

def get_chat(model_name: str, question: str):
    """chat template for LLMs"""
    prompt = INSTRUCT.format(question)
    if "llama" in model_name:
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    elif "mistral" in model_name:
        chat = [
            {"role": "user", "content": prompt},
        ]
    elif "gemma" in model_name:
        chat = [
            {"role": "user", "content": prompt},
        ]
    elif "qwen" in model_name:
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    else:
        chat = [
            {"role": "user", "content": prompt},
        ]
        
    return chat
    
    
def write_to_csv(generated_sentence, label, file_path):
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([generated_sentence, label]) 
        
        
        
def preprocess_tqa(ds):
    """remove the null string in 'correct_answers' and 'incorrect_answers' """
    def remove_empty_answers(example):
        example["correct_answers"] = [answer for answer in example["correct_answers"] if answer.strip()]
        example["incorrect_answers"] = [answer for answer in example["incorrect_answers"] if answer.strip()]
        return example
    
    filtered_ds = ds.map(remove_empty_answers)
    
    return filtered_ds


def preprocess_tqa_mc(ds):
    """Process the multiple-choice dataset of TruthfulQA.
    
    The multiple-choice dataset has a different structure compared to the generation dataset.
    It contains 'mc1_targets' and 'mc2_targets' fields, each with 'choices' and 'labels'.
    This function converts it to a format compatible with the existing pipeline by:
    1. Extracting correct answers (where label=1) and incorrect answers (where label=0) from mc1_targets
    2. Creating a new dataset with the same structure as the generation dataset
    3. Preserving the original mc1_targets and mc2_targets for reference
    
    Args:
        ds: The TruthfulQA multiple-choice dataset
        
    Returns:
        A processed dataset with 'question', 'correct_answers', and 'incorrect_answers' fields
        that is compatible with the existing TruthFlow pipeline
    """
    def process_mc_example(example):
        # Extract correct and incorrect answers from mc1_targets
        # mc1_targets has binary labels (1 for correct, 0 for incorrect)
        correct_answers = []
        incorrect_answers = []
        
        for choice, label in zip(example["mc1_targets"]["choices"], example["mc1_targets"]["labels"]):
            if label == 1:
                correct_answers.append(choice)
            else:
                incorrect_answers.append(choice)
        
        # Create a new example with the same structure as the generation dataset
        return {
            "question": example["question"],
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
            # Keep the original mc targets for reference if needed
            "mc1_targets": example["mc1_targets"],
            "mc2_targets": example["mc2_targets"]
        }
    
    processed_ds = ds.map(process_mc_example)
    
    return processed_ds

