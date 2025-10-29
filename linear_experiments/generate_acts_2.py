import torch
import torch as t
# MODIFIED: Import StoppingCriteria for robust stopping
from transformers import (AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer,
                          LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList)
import argparse
import pandas as pd
from tqdm import tqdm
import os
import glob
from thefuzz import process, fuzz

# --- NEW: Custom StoppingCriteria Class ---
class StopOnTokens(StoppingCriteria):
    """
    A custom StoppingCriteria that stops generation when any of the specified
    stop token IDs are generated.
    """
    def __init__(self, stop_ids: list):
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if the last generated token is in the stop list
        if input_ids[0][-1] in self.stop_ids:
            return True
        return False

# --- User-provided helper functions ---
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

def tokenize(prompt, tokenizer, model_name, tokenizer_args=None):
    if 'instruct' in model_name.lower() or 'it' in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        model_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda')
    else:
        model_input = tokenizer(prompt, return_tensors='pt', **(tokenizer_args or {})).to('cuda')
        if "input_ids" in model_input:
            model_input = model_input["input_ids"]
    return model_input

def create_prompts(statements, model_name):
    mn_lower = model_name.lower()
    
    if 'instruct' in mn_lower or 'it' in mn_lower:
        return statements
    elif 'gemma' in mn_lower:
        return [f"<start_of_turn>user\nQ: {s}<end_of_turn>\n<start_of_turn>model\nA:" for s in statements]
    else:
        return [f"Q:<start_of_turn>user\n{s}<end_of_turn>\n<start_of_turn>model\nA:" for s in statements]


def generate_model_answers(data, model, tokenizer, device, model_name, do_sample=False,
                           temperature=1.0, top_p=1.0, max_new_tokens=100, stop_token_id=None, verbose=False,
                           stopping_criteria=None):
    all_textual_answers = []
    all_input_output_ids = []
    
    for prompt in data:
        model_input = tokenize(prompt, tokenizer, model_name)
        with torch.no_grad():
            model_output = generate(model_input, model, model_name, do_sample=do_sample,
                                    max_new_tokens=max_new_tokens,
                                    top_p=top_p, temperature=temperature,
                                    stop_token_id=stop_token_id, tokenizer=tokenizer,
                                    stopping_criteria=stopping_criteria)
        
        answer = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):], skip_special_tokens=True)
        all_textual_answers.append(answer)
        all_input_output_ids.append(model_output['sequences'][0].cpu())
    return all_textual_answers, all_input_output_ids

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

import re

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


def extract_answer_with_llm(question, model_answer, model, tokenizer, max_retries=2):
    """
    Hybrid approach: Try direct extraction first, then conservative LLM extraction
    """
    
    # First, try direct pattern-based extraction
    direct_answer = extract_answer_direct(model_answer, question)
    if direct_answer and not is_vague_or_non_answer(direct_answer, question):
        return direct_answer
    
    # If direct extraction fails, check if the original answer is vague
    if is_vague_or_non_answer(model_answer, question):
        return "NO ANSWER"
    
    # Try LLM extraction as a fallback with simpler prompts
    model_name = model.name_or_path if hasattr(model, 'name_or_path') else 'unknown'
    mn_lower = model_name.lower()
    
    # Simpler, more direct prompts
    prompts_to_try = [
        f"Extract the main answer from this text in 1-5 words:\n{model_answer}\n\nAnswer:",
        f"What is the answer?\n{model_answer}\n\nAnswer:",
        f"Text: {model_answer}\n\nMain answer (brief):"
    ]
    
    for attempt, base_prompt in enumerate(prompts_to_try):
        try:
            # Format according to model type
            if 'instruct' in mn_lower or 'it' in mn_lower:
                messages = [{"role": "user", "content": base_prompt}]
                inputs = tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    return_tensors="pt"
                ).to(model.device)
                inputs = {"input_ids": inputs}
            else:
                if 'gemma' in mn_lower:
                    formatted_prompt = f"<start_of_turn>user\n{base_prompt}<end_of_turn>\n<start_of_turn>model\n"
                else:
                    formatted_prompt = base_prompt
                
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            # Create stopping criteria
            stop_tokens = ['<end_of_turn>', '<eos>', '</s>', '<|eot_id|>', '\n']
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
                    **inputs,
                    max_new_tokens=15,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                    repetition_penalty=1.2
                )
            
            # Decode only new tokens
            new_tokens = outputs[0, inputs['input_ids'].shape[1]:]
            decoded_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up the answer using the same direct extraction logic
            cleaned_answer = extract_answer_direct(decoded_answer, question)
            
            # Validate the answer much more strictly
            if cleaned_answer and len(cleaned_answer) > 1:
                # First check if the cleaned answer itself is vague
                if is_vague_or_non_answer(cleaned_answer, question):
                    continue  # Try next attempt
                
                # Check if it's a reasonable extraction
                if (len(cleaned_answer.split()) <= 6 and  # Allow up to 6 words
                    not any(phrase in cleaned_answer.lower() for phrase in [
                        "extract", "answer from", "question", "response", "text:", "main", "brief"
                    ]) and
                    not cleaned_answer.startswith('Q:') and
                    not cleaned_answer.startswith('A:') and
                    not cleaned_answer.startswith('The ') and  # Often indicates incomplete extraction
                    not cleaned_answer.startswith('It ') and   # Often indicates incomplete extraction
                    not cleaned_answer.startswith('This ') and # Often indicates incomplete extraction
                    not cleaned_answer.startswith('That ')):   # Often indicates incomplete extraction
                    
                    # Check if answer words appear in original response (stricter)
                    answer_words = set(cleaned_answer.lower().split())
                    response_words = set(model_answer.lower().split())
                    
                    # Require at least 60% of answer words to be in original response
                    if len(answer_words) > 0:
                        overlap = len(answer_words.intersection(response_words))
                        overlap_ratio = overlap / len(answer_words)
                        
                        if overlap_ratio >= 0.6:  # At least 60% overlap
                            return cleaned_answer
            
        except Exception as e:
            print(f"LLM extraction attempt {attempt + 1} failed: {e}")
            continue
    
    # If all attempts fail, return NO ANSWER
    return "NO ANSWER"


def _cleanup_extracted_answer(decoded_output):
    """
    Simplified cleanup function - just use the direct extraction
    """
    return extract_answer_direct(decoded_output, "")

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

class Hook:
    def __init__(self): self.out = None
    def __call__(self, module, module_inputs, module_outputs):
        self.out = module_outputs[0] if isinstance(module_outputs, tuple) else module_outputs

def load_model(model_repo_id: str, device: str):
    print(f"Loading from Hugging Face Hub: {model_repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_repo_id,
        device_map='auto',
        torch_dtype=t.bfloat16 if t.cuda.is_available() and t.cuda.is_bf16_supported() else t.float16
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    layers = getattr(getattr(model, 'model', None), 'layers', None)
    if layers is None:
        raise AttributeError(f"Could not find layers for model {model_repo_id}. Please check the model architecture.")
    return tokenizer, model, layers

# FIXED: Added encoding='utf-8' for robust file reading.
def load_statements(dataset_name):
    path = f"datasets/{dataset_name}.csv"
    df = pd.read_csv(path, encoding='utf-8')
    question_col = 'statement' if 'statement' in df.columns else 'raw_question'
    label_col = 'label' if 'label' in df.columns else 'correct_answer'
    if question_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Dataset {dataset_name}.csv must have a question and an answer column.")
    return df, df[question_col].tolist(), df[label_col].tolist()

def get_acts(statements, correct_answers, tokenizer, model, layers, layer_indices, device, enable_llm_extraction=False):
    model_name = model.name_or_path if hasattr(model, 'name_or_path') else 'unknown'
    
    attn_hooks, mlp_hooks = {}, {}
    handles = []
    for l in layer_indices: 
        hook_a, hook_m = Hook(), Hook()
        handles.extend([
            layers[l].self_attn.register_forward_hook(hook_a),
            layers[l].mlp.register_forward_hook(hook_m)
        ])
        attn_hooks[l], mlp_hooks[l] = hook_a, hook_m

    prompts = create_prompts(statements, model_name)
    
    stop_tokens = ['\n', '<end_of_turn>', '<eos>']
    stop_ids = set([tokenizer.encode(st, add_special_tokens=False)[-1] for st in stop_tokens])
    stopping_criteria = StoppingCriteriaList([StopOnTokens(list(stop_ids))])
    
    all_model_answers_raw, all_generated_ids = generate_model_answers(
        prompts, model, tokenizer, device, model_name, max_new_tokens=64,
        stopping_criteria=stopping_criteria
    )

    acts = {2*l: [] for l in layer_indices}
    acts.update({2*l + 1: [] for l in layer_indices})
    batch_correctness, batch_model_answers, batch_exact_answers = [], [], []

    iterator = zip(statements, prompts, correct_answers, all_model_answers_raw, all_generated_ids)
    for stmt, prompt, correct_ans, model_answer_text_raw, generated_ids in tqdm(iterator, desc="Processing batch", total=len(statements), leave=False):
        
        model_answer_text = model_answer_text_raw.strip()
        
        batch_model_answers.append(model_answer_text)
        is_correct = check_correctness(model_answer_text, correct_ans)
        batch_correctness.append(is_correct)
        
        exact_answer_str = find_exact_answer_simple(model_answer_text, correct_ans)
        
        # Use the improved LLM extraction
        if not exact_answer_str and enable_llm_extraction:
            print(f"Trying LLM extraction for: {stmt[:50]}...")
            exact_answer_str = extract_answer_with_llm(stmt, model_answer_text, model, tokenizer)
            if exact_answer_str:
                print(f"LLM extracted: '{exact_answer_str}'")
        
        batch_exact_answers.append(exact_answer_str)
        
        if not exact_answer_str:
            continue

        input_ids = tokenize(prompt, tokenizer, model_name)
        answer_token_indices = find_answer_token_indices_by_string_matching(tokenizer, generated_ids, input_ids, exact_answer_str)
        if answer_token_indices is None:
            continue
        
        with t.no_grad():
            model(generated_ids.unsqueeze(0).to(device))

        for l in layer_indices:
            try:
                a = attn_hooks[l].out[0][answer_token_indices].mean(dim=0).detach()
                m = mlp_hooks[l].out[0][answer_token_indices].mean(dim=0).detach()
                acts[2*l].append(a)
                acts[2*l+1].append(m)
            except IndexError:
                print(f"IndexError on layer {l}. Skipping activation extraction for this item.")
                break

    for k in list(acts.keys()):
        if acts[k]: acts[k] = t.stack(acts[k]).cpu().float()
        else: del acts[k]

    for h in handles: h.remove()
        
    return acts, batch_correctness, batch_model_answers, batch_exact_answers

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract activations and check correctness from Hugging Face models.")
    parser.add_argument('--model_repo_id', type=str, required=True, help='The exact Hugging Face Hub repository ID (e.g., google/gemma-2b-it)')
    parser.add_argument('--layers', nargs='+', required=True, type=int, help='Layer indices to extract from (-1 for all)')
    parser.add_argument('--datasets', nargs='+', required=True, help='Dataset names (without .csv)')
    parser.add_argument('--output_dir', default='acts_output', help='Root directory for saving all outputs')
    parser.add_argument('--device', default='cuda' if t.cuda.is_available() else 'cpu', help='Device to run on (cpu or cuda)')
    parser.add_argument('--enable_llm_extraction', action='store_true', help='Enable LLM-based answer extraction as a fallback.')
    parser.add_argument('--early_stop', action='store_true', help='Process only two batches and save a subsampled CSV for quick checks.')
    args = parser.parse_args()

    ds = args.datasets
    if ds == ['all']:
        ds = [os.path.relpath(fp, 'datasets').replace('.csv', '') for fp in glob.glob('datasets/**/*.csv', recursive=True)]

    t.set_grad_enabled(False)
    tokenizer, model, layer_modules = load_model(args.model_repo_id, args.device)
    
    li = args.layers
    if -1 in li: li = list(range(len(layer_modules)))

    for dataset in ds:
        print(f"\n--- Processing dataset: {dataset} ---")
        try:
            df, stmts, correct_answers = load_statements(dataset)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping dataset '{dataset}': {e}")
            continue
            
        safe_repo_name = args.model_repo_id.replace("/", "__")
        save_base = os.path.join(args.output_dir, safe_repo_name, dataset)
        os.makedirs(save_base, exist_ok=True)
        
        all_correctness_results, all_model_answers, all_exact_answers = [], [], []
        
        batch_size = 25
        batch_count = 0 
        for start in tqdm(range(0, len(stmts), batch_size), desc=f"Overall progress for {dataset}"):
            batch_stmts = stmts[start:start + batch_size]
            batch_correct_ans = correct_answers[start:start + batch_size]
            
            acts, batch_correctness, batch_model_ans, batch_exact_ans = get_acts(
                batch_stmts, batch_correct_ans, tokenizer, model, layer_modules, li, args.device, 
                enable_llm_extraction=args.enable_llm_extraction
            )
            
            all_correctness_results.extend(batch_correctness)
            all_model_answers.extend(batch_model_ans)
            all_exact_answers.extend(batch_exact_ans)

            if acts:
                for pseudo, tensor in acts.items():
                    filename = os.path.join(save_base, f"layer_{pseudo}_{start}.pt")
                    t.save(tensor, filename)
            
            batch_count += 1
            if args.early_stop and batch_count >= 2:
                print(f"\nEarly stopping after {batch_count} batches.")
                break
        
        num_results = len(all_model_answers)
        if args.early_stop:
            if num_results > 0:
                df_sub = df.iloc[:num_results].copy()
                df_sub['model_answer'] = all_model_answers
                df_sub['automatic_correctness'] = all_correctness_results
                df_sub['exact_answer'] = all_exact_answers
                
                output_csv_path = os.path.join(save_base, f"{dataset}_SUBSAMPLED_with_results.csv")
                # FIXED: Added encoding='utf-8' for robust file writing.
                df_sub.to_csv(output_csv_path, index=False, encoding='utf-8')
                print(f"✅ Early stop: Saved subsampled results to: {output_csv_path}")
            else:
                print("⚠️ Warning: No results generated during early stop run. No CSV saved.")
        elif num_results == len(df):
            df['model_answer'] = all_model_answers
            df['automatic_correctness'] = all_correctness_results
            df['exact_answer'] = all_exact_answers
            
            output_csv_path = os.path.join(save_base, f"{dataset}_with_results.csv")
            # FIXED: Added encoding='utf-8' for robust file writing.
            df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"✅ Saved full dataset with results to: {output_csv_path}")
        else:
            print(f"⚠️ Warning: Mismatch between results ({num_results}) and dataset rows ({len(df)}). CSV not saved.")