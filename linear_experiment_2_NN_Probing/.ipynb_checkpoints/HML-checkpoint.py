import os
import json
import numpy as np

import os
import glob
import torch as t
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from classifier import ProbingNetwork, hparams
from collections import defaultdict
def compute_generation_accuracy(output_dir="validation"):
    """
    Loads the generations cache for the given model and computes:
      - Per-statement accuracy (fraction of correct generations out of 32)
      - Overall accuracy across all statements
      - Category list: H (High >0.75), L (Low <0.25), M (Medium otherwise)
    """
    model_name = model_name.replace("/", "_")
    generations_dir = os.path.join(output_dir, "generations")
    generations_cache_path = os.path.join(generations_dir, "generated_validation_set.json")

    if not os.path.exists(generations_cache_path):
        raise FileNotFoundError(f"No generations cache found at {generations_cache_path}")

    with open(generations_cache_path, 'r', encoding='utf-8') as f:
        generations_cache = json.load(f)

    per_statement_acc = {}
    per_statement_category = {}
    all_labels = []
    HML=defaultdict(list)

    for stmt, data in generations_cache.items():
        labels = data.get("ground_truth_labels", [])
        if not labels:
            continue
        acc = np.mean(labels)
        HML[stmt].append(acc)


        # categorize
        if acc > 0.75:
            cat = "H"
        elif acc < 0.25:
            cat = "L"
        else:
            cat = "M"
        HML[stmt].append(cat)

        all_labels.extend(labels)

    overall_acc = np.mean(all_labels) if all_labels else 0.0

    print(f"Evaluation complete.\nOverall accuracy: {overall_acc:.4f}")
    return HML

def evaluate_probe_per_statement(home_dir, eval_layers, device='cuda'):
    """
    For each layer:
      - Loads the trained probe
      - Evaluates per-statement accuracy (on 64 generations TRUE/FALSE)
    Returns:
      results[layer][stmt_id] = accuracy
    """
    model_name_safe = hparams.model_name.replace('/', '_')
    projected_dir = os.path.join(home_dir, 'activations/activations_svd_val', model_name_safe)
    probes_dir = os.path.join(home_dir, 'trained_probes', model_name_safe)

    results = {}

    for l_idx in tqdm(eval_layers, desc="Evaluating probes per layer"):
        # Load probe
        model_path = os.path.join(probes_dir, f'probe_model_layer_{l_idx}.pt')
        if not os.path.exists(model_path):
            print(f"No trained probe found for layer {l_idx}, skipping.")
            continue

        model = ProbingNetwork(hparams.model_name).to(device)
        model.load_state_dict(t.load(model_path, map_location=device))
        model.eval()

        # For each statement file
        stmt_results = {}
        file_pattern = os.path.join(projected_dir, f'layer{l_idx}_stmt*_svd_processed.pt')
        files = sorted(glob.glob(file_pattern))

        for fname in tqdm(files, desc=f"L{l_idx} per-stmt", leave=False):
            data = t.load(fname, map_location=device)
            X, y = data['activations'].to(device).float(), data['labels'].float().to(device)

            with t.no_grad():
                outputs = model(X)
                preds = (outputs > 0.5).float()

            acc = accuracy_score(y.cpu().numpy(), preds.cpu().numpy())

            # Extract statement id from filename
            stmt_id = os.path.basename(fname).split("_stmt_")[-1].split("_")[0]
            stmt_results[stmt_id] = acc

        results[l_idx] = stmt_results
        print(f"Layer {l_idx}: evaluated {len(stmt_results)} statements.")

    return results


def runHML(HML_out_dir,network_in_dir,network_out_dir,eval_layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],gen_in_dir='current_run'):
    HML = compute_generation_accuracy(gen_in_dir)
    with open(f"{HML_out_dir}/HML_gen.json", "w", encoding="utf-8") as f:
        json.dump(HML, f, indent=4)
    print(f"Results saved to {HML_out_dir}/HML_gen.json")
    
    classifier_acc=evaluate_probe_per_statement(network_in_dir, eval_layers)

    with open(f"{network_out_dir}/net_acc.json", "w", encoding="utf-8") as f:
        json.dump(classifier_acc, f, indent=4)
    print(f"Results saved to {network_out_dir}/net_acc.json")
    






    
