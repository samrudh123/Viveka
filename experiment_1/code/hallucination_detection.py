import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# You will need these utility functions from your project
from probing_utils import extract_internal_reps_specific_layer_and_token, load_model_and_validate_gpu

import argparse

# --- Step 1: Parse Arguments ---
parser = argparse.ArgumentParser(description="Run probing experiment for Gemma-2B")

parser.add_argument('--pkl_path', type=str, required=True,
                    help='Relative path to the .pkl file under ../checkpoints/')
parser.add_argument('--probe_token', type=str, required=True,
                    help='Token to probe in the input sequence')
parser.add_argument('--probe_at', type=str, required=True,
                    help='Component to probe at (e.g., attention_output, mlp, etc.)')
parser.add_argument('--probe_layer', type=int, default=24,
                    help='Layer to probe at (default: 24)')
parser.add_argument('--model_name', type=str, default='google/gemma-2-2b-it',
                    help='Name or path of the Hugging Face model (default: google/gemma-2-2b-it)')

args = parser.parse_args()

# --- Step 2: Assign Parameters ---
MODEL_NAME = args.model_name
PKL_FILE_PATH = f"../checkpoints/{args.pkl_path}"
PROBE_LAYER = args.probe_layer
PROBE_TOKEN = args.probe_token
PROBE_AT = args.probe_at

print("Loading the base language model (e.g., Gemma)...")
model, tokenizer = load_model_and_validate_gpu(MODEL_NAME)

print(f"Loading the probe from {PKL_FILE_PATH}...")
with open(PKL_FILE_PATH, 'rb') as f:
    probe_clf = pickle.load(f)

import torch
import numpy as np

import torch
import numpy as np

# --- Step 1: TriviaQA-style Prompts and Ground Truths ---
questions_list = [
    "What is the capital of France?",
    "Who painted the Mona Lisa?",
    "What is the smallest planet in our solar system?",
    "Which element has the chemical symbol 'O'?",
    "What year did World War II end?",
    "Who discovered penicillin?",
    "What is the tallest mountain in the world?",
    "In which country is the Great Pyramid of Giza located?",
    "Who wrote 'Pride and Prejudice'?",
    "What is the square root of 144?"
]

true_answers_list = [
    "Paris",
    "Leonardo da Vinci",
    "Mercury",
    "Oxygen",
    "1945",
    "Alexander Fleming",
    "Mount Everest",
    "Egypt",
    "Jane Austen",
    "12"
]

# --- Step 2: Generate Model Answers ---
print("Generating model answers using Gemma...")
model_answers_list = []

for question in questions_list:
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            do_sample=False
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_answer = output_text.replace(question, "").strip()
    model_answers_list.append(generated_answer)

# --- Step 3: Tokenize for Probing ---
input_output_ids = []

for question, answer in zip(questions_list, model_answers_list):
    q_ids = tokenizer([question], return_tensors='pt').input_ids[0]
    a_ids = tokenizer([answer], return_tensors='pt').input_ids[0]
    combined_ids = torch.cat((q_ids, a_ids[1:]), dim=0)
    input_output_ids.append(combined_ids)

dummy_labels = [1] * len(questions_list)

# --- Step 4: Extract Hidden States ---
hidden_vectors = extract_internal_reps_specific_layer_and_token(
    model,
    tokenizer,
    questions_list,
    input_output_ids,
    PROBE_AT,
    MODEL_NAME,
    PROBE_LAYER,
    PROBE_TOKEN,
    model_answers_list,
    dummy_labels
)

X_new = np.array(hidden_vectors)

# --- Step 5: Run the Probe ---
predicted_classes = probe_clf.predict(X_new)
probabilities = probe_clf.predict_proba(X_new)

# --- Step 6: Display Results ---
print("\n--- Inference Results ---")
for i in range(len(questions_list)):
    print(f"\nQ{i+1}: {questions_list[i]}")
    print(f"✅ Correct Answer: {true_answers_list[i]}")
    print(f"🧠 Model's Answer: {model_answers_list[i]}")
    print(f"🔍 Probe Prediction: {'NOT a hallucination' if predicted_classes[i] == 1 else 'HALLUCINATION'}")
    print(f"📊 Confidence in correctness: {probabilities[i][1]:.2%}")