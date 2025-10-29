import json
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch as t
import transformers
from baukit import TraceDict
from datasets import load_dataset
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer,
                          LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList)
import argparse
import os
import glob
from thefuzz import process, fuzz
import re

N_LAYER_GEMMA_2 = 26

LAYERS_TO_TRACE_GEMMA_2 = {
    'mlp': [f"model.layers.{i}.mlp" for i in range(N_LAYER_GEMMA_2)],
    'mlp_last_layer_only': [f"model.layers.{i}.mlp.down_proj" for i in range(N_LAYER_GEMMA_2)],
    'mlp_last_layer_only_input': [f"model.layers.{i}.mlp.down_proj" for i in range(N_LAYER_GEMMA_2)],
    'attention_heads': [f"model.layers.{i}.self_attn.o_proj" for i in range(N_LAYER_GEMMA_2)],
    'attention_output': [f"model.layers.{i}.self_attn.o_proj" for i in range(N_LAYER_GEMMA_2)],
    'residual_activations': [f"model.layers.{i}.residual_activations" for i in range(N_LAYER_GEMMA_2)],
}

LAYERS_TO_TRACE = {
    "google/gemma-2-2b-it" : LAYERS_TO_TRACE_GEMMA_2,
    "google/gemma-2-2b" : LAYERS_TO_TRACE_GEMMA_2,
}

N_LAYERS = {
    "google/gemma-2-2b-it" : N_LAYER_GEMMA_2,
    'google/gemma-2-2b' : N_LAYER_GEMMA_2,
}

HIDDEN_SIZE = {
    'google/gemma-7b': 3072,
    'google/gemma-7b-it': 3072,
    'google/gemma-2-2b-it' : 2304,
    'google/gemma-2-2b' : 2304,
}

LIST_OF_DATASETS = ['triviaqa',
                    'imdb',
                    'winobias',
                    'hotpotqa',
                    'hotpotqa_with_context',
                    'math',
                    'movies',
                    'mnli',
                    'natural_questions_with_context',
                    'winogrande']

LIST_OF_TEST_DATASETS = [f"{x}_test" for x in LIST_OF_DATASETS]

LIST_OF_MODELS = ["google/gemma-2-2b-it","gemini-2.5-flash","google/gemma-3-4b-it","google/gemma-2-2b"]

MODEL_FRIENDLY_NAMES = {
    'google/gemma-2-2b-it' : 'gemma-2-2b-instruct',
    'google/gemma-2-2b' : 'gemma-2-2b',
}

LIST_OF_PROBING_LOCATIONS = ['mlp', 'mlp_last_layer_only', 'mlp_last_layer_only_input', 'attention_output','resdiual_activations']

class StopOnTokens(StoppingCriteria):
    """
    A custom StoppingCriteria that stops generation when any of the specified
    stop token IDs are generated.
    """
    def __init__(self, stop_ids: list):
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.stop_ids:
            return True
        return False


def encode(prompt, tokenizer, model_name):
    messages = [
        {"role": "user", "content": prompt}
    ]
    model_input = tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
    return model_input

def tokenize(prompt, tokenizer, model_name, tokenizer_args=None):
    """Fixed tokenization function with proper attention mask"""
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        add_special_tokens=True,
        padding=True,
        truncation=True
    )
    return inputs['input_ids'].squeeze(), inputs.get('attention_mask', None)

def generate(model_input, model, model_name, do_sample=False, output_scores=False, temperature=1.0, top_k=50, top_p=1.0,
             max_new_tokens=100, stop_token_id=None, tokenizer=None, output_hidden_states=False, additional_kwargs=None,
             stopping_criteria=None):
    if stop_token_id is not None: eos_token_id = stop_token_id
    else: eos_token_id = None
    
    model_output = model.generate(model_input,
                                  max_new_tokens=max_new_tokens, output_hidden_states=output_hidden_states,
                                  output_scores=output_scores,
                                  return_dict_in_generate=True, do_sample=do_sample,
                                  temperature=temperature, top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
                                  stopping_criteria=stopping_criteria,
                                  **(additional_kwargs or {}))
    return model_output

def create_prompts(statements, model_name):
    """Applies the correct prompt format based on the model type."""
    mn_lower = model_name.lower()
    
    if 'instruct' in mn_lower or 'it' in mn_lower:
        return [f"<start_of_turn>user\nQ: {s}<end_of_turn>\n<start_of_turn>model\nA:" for s in statements]
    else:
        return statements

def generate_model_answers(
    prompts,
    model,
    tokenizer,
    device,
    model_name,
    max_tokens=2,
    stopping_criteria=None,
    num_return_sequences=1,
    temperature = 0.7,
    top_p = 0.9,
    do_sample = True,
    additional_kwargs = None
    ):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            model.resize_token_embeddings(len(tokenizer))

    # Tokenize all prompts at once
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True
    ).to(device)

    # Generate in batch
    print(max_tokens, " : Max tokens at utils.py, generate_model_answers")
    with t.no_grad():
        generated = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            **(additional_kwargs or {})  # To handle exceptions of unexpected args
        )

    # Decode outputs
    all_model_answers_raw = []
    all_generated_ids = []

    batch_size = len(prompts)
    for i in range(batch_size * num_return_sequences):
        prompt_idx = i // num_return_sequences
        input_len = inputs['input_ids'][prompt_idx].shape[0]

        generated_ids = generated[i][input_len:]
        model_answer_raw = tokenizer.decode(generated_ids, skip_special_tokens=True)

        all_model_answers_raw.append(model_answer_raw)
        all_generated_ids.append(generated_ids.cpu())

    return all_model_answers_raw, all_generated_ids


def check_correctness(model_answer, correct_answers_list):
    if isinstance(correct_answers_list, str):
        try: labels_ = eval(correct_answers_list)
        except: labels_ = [correct_answers_list]
    else: labels_ = correct_answers_list
    if not isinstance(labels_, list): labels_ = [str(labels_)]
    if not isinstance(model_answer, str): return 0
    for ans in labels_:
        if str(ans).lower() in model_answer.lower(): return 1
    return 0

def find_exact_answer_simple(model_answer: str, correct_answer):
    if not isinstance(model_answer, str): return None
    try:
        if isinstance(correct_answer, str):
            correct_answer_eval = eval(correct_answer)
            if isinstance(correct_answer_eval, list): correct_answer = correct_answer_eval
    except (SyntaxError, NameError): pass
    found_ans, found_ans_index = "", -1
    if isinstance(correct_answer, list):
        current_best_index = len(model_answer)
        for ans in correct_answer:
            ans_str = str(ans)
            ans_index = model_answer.lower().find(ans_str.lower())
            if ans_index != -1 and ans_index < current_best_index:
                found_ans, current_best_index = ans_str, ans_index
        found_ans_index = current_best_index if found_ans else -1
    else:
        ans_str = str(correct_answer)   
        found_ans_index = model_answer.lower().find(ans_str.lower())
        found_ans = ans_str
    if found_ans_index != -1:
        return model_answer[found_ans_index : found_ans_index + len(found_ans)]
    return None


def extract_answer_direct(model_answer, question):
    """
    Direct extraction using patterns - much more reliable than LLM extraction
    """
    if not model_answer or len(model_answer.strip()) == 0:
        return None
    
    answer = model_answer.strip()
    
    # Remove common prefixes that don't contain the answer
    prefixes_to_remove = [
        "The answer is ",
        "The correct answer is ",
        "It is ",
        "This is ",
        "That is ",
        "It was ",
        "This was ",
        "That was "
    ]
    
    for prefix in prefixes_to_remove:
        if answer.startswith(prefix):
            answer = answer[len(prefix):]
            break
    
    # Stop at common sentence endings
    stop_patterns = [
        r'\.',  # Period
        r'\n',  # Newline
        r' is ',  # " is "
        r' was ',  # " was "
        r' are ',  # " are "
        r' were ',  # " were "
        r' which ',  # " which "
        r' who ',  # " who "
        r' that ',  # " that "
        r' and ',  # " and "
        r' but ',  # " but "
        r' however ',  # " however "
        r' although ',  # " although "
        r' because ',  # " because "
        r' since ',  # " since "
        r' while ',  # " while "
        r' during ',  # " during "
        r' after ',  # " after "
        r' before ',  # " before "
        r' in \d{4}',  # " in 1985" (year)
        r' from \d{4}',  # " from 1985"
        r' between \d{4}',  # " between 1985"
    ]
    
    for pattern in stop_patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            answer = answer[:match.start()].strip()
            break
    
    # Clean up the answer
    answer = answer.strip()
    
    # Remove quotes
    answer = answer.strip('"\'')
    
    # Remove trailing punctuation
    answer = answer.rstrip('.,!?;:')
    
    # If it's too long, take first few words (but be less aggressive)
    words = answer.split()
    if len(words) > 8:  # Increased from 5 to 8
        answer = " ".join(words[:5])  # Take first 5 words instead of 3
    
    return answer if answer and len(answer) > 1 else None


def is_vague_or_non_answer(model_answer, question):
    """
    Check if the model answer is vague or doesn't properly answer the question
    Much more comprehensive detection
    """
    if not model_answer or len(model_answer.strip()) == 0:
        return True
    
    answer_lower = model_answer.lower().strip()
    
    # Check for explicit non-answer patterns
    non_answer_patterns = [
        "i don't know",
        "i'm not sure",
        "i cannot",
        "i can't",
        "unable to",
        "don't have information",
        "not available",
        "not provided",
        "i'm sorry",
        "i apologize",
        "cannot determine",
        "cannot provide",
        "not specified",
        "not mentioned",
        "not clear",
        "difficult to determine",
        "hard to determine",
        "no information",
        "no details",
        "cannot answer",
        "not sure",
        "unclear",
        "unknown",
        "not known",
        "can't say",
        "cannot say",
        "not certain",
        "uncertain"
    ]
    
    for pattern in non_answer_patterns:
        if pattern in answer_lower:
            return True
    
    # Check for incomplete/cut-off sentences that suggest the model didn't finish properly
    incomplete_patterns = [
        "the word",
        "the opening",
        "the first",
        "the last",
        "the main",
        "the original",
        "the story",
        "the film",
        "the movie",
        "the book",
        "the song",
        "the album",
        "the show",
        "the series",
        "the episode",
        "the character",
        "the actor",
        "the actress",
        "the director",
        "the author",
        "the artist",
        "the band",
        "the group",
        "the title",
        "the name",
        "the question",
        "the answer",
        "based on",
        "according to",
        "refers to",
        "relates to",
        "is about",
        "deals with",
        "concerns",
        "involves",
        "features",
        "includes",
        "contains",
        "describes",
        "explains",
        "discusses",
        "mentions",
        "states",
        "says",
        "tells",
        "shows",
        "indicates",
        "suggests",
        "implies",
        "means",
        "represents"
    ]
    
    # Check if answer starts with these incomplete patterns
    for pattern in incomplete_patterns:
        if answer_lower.startswith(pattern):
            return True
    
    # Check if answer ends abruptly (incomplete sentence indicators)
    if (answer_lower.endswith(" is") or 
        answer_lower.endswith(" was") or 
        answer_lower.endswith(" are") or 
        answer_lower.endswith(" were") or
        answer_lower.endswith(" the") or 
        answer_lower.endswith(" a") or 
        answer_lower.endswith(" an") or
        answer_lower.endswith(" of") or
        answer_lower.endswith(" in") or
        answer_lower.endswith(" on") or
        answer_lower.endswith(" at") or
        answer_lower.endswith(" to") or
        answer_lower.endswith(" for") or
        answer_lower.endswith(" with") or
        answer_lower.endswith(" by") or
        answer_lower.endswith(" from") or
        answer_lower.endswith(" about") or
        answer_lower.endswith(" that") or
        answer_lower.endswith(" which") or
        answer_lower.endswith(" who") or
        answer_lower.endswith(" when") or
        answer_lower.endswith(" where") or
        answer_lower.endswith(" how") or
        answer_lower.endswith(" why")):
        return True
    
    # Check for overly generic answers
    generic_answers = [
        "it depends",
        "various",
        "many",
        "several",
        "different",
        "multiple",
        "numerous",
        "some",
        "there are many",
        "there are several",
        "there are various",
        "it varies",
        "varies",
        "depends on",
        "based on the",
        "according to the",
        "in the context",
        "contextual",
        "context-dependent",
        "situational",
        "case-by-case",
        "depends on context",
        "depends on the situation"
    ]
    
    for pattern in generic_answers:
        if pattern in answer_lower:
            return True
    
    # Check if answer is too short (less than 2 characters)
    if len(answer_lower) < 2:
        return True
    
    # Check if answer is just common words
    if answer_lower in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'out', 'down', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now']:
        return True
    
    return False


# def extract_answer_with_llm(question, model_answer, model, tokenizer, max_retries=2):
#     """
#     Hybrid approach: Try direct extraction first, then conservative LLM extraction
#     """
    
#     # First, try direct pattern-based extraction
#     direct_answer = extract_answer_direct(model_answer, question)
#     if direct_answer and not is_vague_or_non_answer(direct_answer, question):
#         return direct_answer
    
#     # If direct extraction fails, check if the original answer is vague
#     if is_vague_or_non_answer(model_answer, question):
#         return "NO ANSWER"
    
#     # Try LLM extraction as a fallback with simpler prompts
#     model_name = model.name_or_path if hasattr(model, 'name_or_path') else 'unknown'
#     mn_lower = model_name.lower()
    
#     # Simpler, more direct prompts
#     prompts_to_try = [
#         f"Extract the main answer from this text in 1-5 words:\n{model_answer}\n\nAnswer:",
#         f"What is the answer?\n{model_answer}\n\nAnswer:",
#         f"Text: {model_answer}\n\nMain answer (brief):"
#     ]
    
#     for attempt, base_prompt in enumerate(prompts_to_try):
#         try:
#             # Format according to model type
#             if 'instruct' in mn_lower or 'it' in mn_lower:
#                 messages = [{"role": "user", "content": base_prompt}]
#                 inputs = tokenizer.apply_chat_template(
#                     messages, 
#                     add_generation_prompt=True, 
#                     return_tensors="pt"
#                 ).to(model.device)
#                 inputs = {"input_ids": inputs}
#             else:
#                 if 'gemma' in mn_lower:
#                     formatted_prompt = f"<start_of_turn>user\n{base_prompt}<end_of_turn>\n<start_of_turn>model\n"
#                 else:
#                     formatted_prompt = base_prompt
                
#                 inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
#             # Create stopping criteria
#             stop_tokens = ['<end_of_turn>', '<eos>', '</s>', '<|eot_id|>', '\n']
#             stop_ids = []
#             for st in stop_tokens:
#                 try:
#                     encoded = tokenizer.encode(st, add_special_tokens=False)
#                     if encoded:
#                         stop_ids.append(encoded[-1])
#                 except:
#                     continue
            
#             stopping_criteria = StoppingCriteriaList([StopOnTokens(list(set(stop_ids)))])
            
#             # Generate with conservative parameters
#             with t.no_grad():
#                 outputs = model.generate(
#                     **inputs,
#                     max_new_tokens=15,
#                     temperature=0.1,
#                     do_sample=True,
#                     pad_token_id=tokenizer.eos_token_id,
#                     stopping_criteria=stopping_criteria,
#                     repetition_penalty=1.2
#                 )
            
#             # Decode only new tokens
#             new_tokens = outputs[0, inputs['input_ids'].shape[1]:]
#             decoded_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
#             # Clean up the answer using the same direct extraction logic
#             cleaned_answer = extract_answer_direct(decoded_answer, question)
            
#             # Validate the answer much more strictly
#             if cleaned_answer and len(cleaned_answer) > 1:
#                 # First check if the cleaned answer itself is vague
#                 if is_vague_or_non_answer(cleaned_answer, question):
#                     continue  # Try next attempt
                
#                 # Check if it's a reasonable extraction
#                 if (len(cleaned_answer.split()) <= 6 and  # Allow up to 6 words
#                     not any(phrase in cleaned_answer.lower() for phrase in [
#                         "extract", "answer from", "question", "response", "text:", "main", "brief"
#                     ]) and
#                     not cleaned_answer.startswith('Q:') and
#                     not cleaned_answer.startswith('A:') and
#                     not cleaned_answer.startswith('The ') and  # Often indicates incomplete extraction
#                     not cleaned_answer.startswith('It ') and   # Often indicates incomplete extraction
#                     not cleaned_answer.startswith('This ') and # Often indicates incomplete extraction
#                     not cleaned_answer.startswith('That ')):   # Often indicates incomplete extraction
                    
#                     # Check if answer words appear in original response (stricter)
#                     answer_words = set(cleaned_answer.lower().split())
#                     response_words = set(model_answer.lower().split())
                    
#                     # Require at least 60% of answer words to be in original response
#                     if len(answer_words) > 0:
#                         overlap = len(answer_words.intersection(response_words))
#                         overlap_ratio = overlap / len(answer_words)
                        
#                         if overlap_ratio >= 0.6:  # At least 60% overlap
#                             return cleaned_answer
            
#         except Exception as e:
#             print(f"LLM extraction attempt {attempt + 1} failed: {e}")
#             continue
    
#     # If all attempts fail, return NO ANSWER
#     return "NO ANSWER"


def extract_answer_with_llm(question, model_answer, model, tokenizer, max_retries=2):
    """
    Simplified approach: Try LLM extraction first, then direct extraction as fallback
    """
    
    # First, try LLM extraction
    llm_answer = try_llm_extraction(question, model_answer, model, tokenizer)
    if llm_answer and llm_answer != "NO ANSWER":
        return llm_answer
    
    # Fallback to direct pattern-based extraction
    direct_answer = extract_answer_direct(model_answer, question)
    if direct_answer and not is_vague_or_non_answer(direct_answer, question):
        return direct_answer
    
    # If both fail, return NO ANSWER
    return "NO ANSWER"


def try_llm_extraction(question, model_answer, model, tokenizer):
    """
    Try to extract answer using the LLM with the specified prompt template
    """
    
    # Create extraction prompt
    extraction_prompt = f"Extract the exact answer from this response. Give only the answer, no explanation.\n\nQuestion: {question}\nResponse: {model_answer}\n\nAnswer:"
    
    # Format using the specified template
    formatted_prompt = f"<start_of_turn>user\n{extraction_prompt}<end_of_turn>\n<start_of_turn>model\nA:"
    
    try:
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Create stopping criteria
        stop_tokens = ['<end_of_turn>', '<eos>', '</s>', '\n']
        stop_ids = []
        for st in stop_tokens:
            try:
                encoded = tokenizer.encode(st, add_special_tokens=False)
                if encoded:
                    stop_ids.append(encoded[-1])
            except:
                continue
        
        stopping_criteria = StoppingCriteriaList([StopOnTokens(list(set(stop_ids)))])
        
        # Generate with conservative parameters
        with t.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=20,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )
        
        # Decode only new tokens
        new_tokens = outputs[0, inputs['input_ids'].shape[1]:]
        decoded_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Basic validation
        if decoded_answer and len(decoded_answer) > 0:
            # Remove common prefixes/artifacts
            cleaned = decoded_answer.strip()
            for prefix in ["A:", "Answer:", "The answer is", "It is"]:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            
            # Basic sanity checks
            if (len(cleaned) > 0 and 
                len(cleaned.split()) <= 10 and  # Reasonable length
                not any(phrase in cleaned.lower() for phrase in [
                    "extract", "response:", "question:", "explanation"
                ])):
                return cleaned
    
    except Exception as e:
        print(f"LLM extraction failed: {e}")
    
    return None

def _cleanup_extracted_answer(decoded_output):
    """
    Simplified cleanup function - just use the direct extraction
    """
    return extract_answer_direct(decoded_output, "")

def load_model(model_repo_id: str, device: str):
    print(f"Loading from Hugging Face Hub: {model_repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_repo_id,
        device_map='auto',
        torch_dtype=t.bfloat16 if t.cuda.is_available() and t.cuda.is_bf16_supported() else t.float16
    )
    print(f"Using dtype:{torch.dtype}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    layers = getattr(getattr(model, 'model', None), 'layers', None)
    if layers is None:
        raise AttributeError(f"Could not find layers for model {model_repo_id}. Please check the model architecture.")
    return tokenizer, model, layers

def load_statements(dataset_name):
    path = f"{dataset_name}"
    df = pd.read_csv(path, encoding='utf-8')
    question_col = 'statement' if 'statement' in df.columns else 'raw_question'
    label_col = 'label' if 'label' in df.columns else 'correct_answer'
    if question_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Dataset {dataset_name}.csv must have a question and an answer column.")
    return df, df[question_col].tolist(), df[label_col].tolist()

def find_answer_token_indices_by_string_matching(tokenizer, full_generated_ids, prompt_ids, exact_answer_str):
    try: full_decoded_text = tokenizer.decode(full_generated_ids, skip_special_tokens=False)
    except: return None
    match = process.extractOne(exact_answer_str, [full_decoded_text], scorer=fuzz.partial_ratio, score_cutoff=90)
    if not match: return None
    best_match_str = match[0]
    start_char = full_decoded_text.find(best_match_str)
    if start_char == -1: return None
    end_char = start_char + len(best_match_str)
    encoding = tokenizer(full_decoded_text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoding['offset_mapping']
    token_indices = []
    for i, (token_start, token_end) in enumerate(offset_mapping):
        if token_start < end_char and start_char < token_end:
            if i < len(full_generated_ids):
                token_indices.append(i)
    return token_indices if token_indices else None