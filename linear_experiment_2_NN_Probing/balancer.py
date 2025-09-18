import os
import random
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import torch
from tqdm import tqdm


def balance_layers_aligned(
    input_dir: str,
    output_dir: str,
    seed: int = 123,
    verbose: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Balance activations across layers, ensuring SAME stmts & pairs are chosen in every layer.
    Also saves JSON file with mapping of chosen stmt files -> chosen indices.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ---- helper: group files by layer ----
    def _group_files_by_layer(inp: str) -> Dict[int, List[str]]:
        layers = defaultdict(list)
        for fname in os.listdir(inp):
            if not fname.endswith(".pt"):
                continue
            parts = fname.split("_")
            if len(parts) < 4:
                continue
            try:
                layer_idx = int(parts[1])
            except Exception:
                continue
            layers[layer_idx].append(os.path.join(inp, fname))
        for k in layers:
            layers[k].sort()
        return dict(sorted(layers.items()))

    # ---- helper: scan a layer and get correct/wrong pairs ----
    def _scan_layer(files: List[str]):
        correct_pairs, wrong_pairs = [], []
        for f_i, fpath in enumerate(files):
            data = torch.load(fpath, map_location="cpu")
            labels = data["labels"]
            if labels.size(0) % 2 != 0:
                labels = labels[:-1]

            ev_lbl = labels[0::2]
            od_lbl = labels[1::2]

            correct_mask = (ev_lbl == 1) & (od_lbl == 0)
            wrong_mask   = (ev_lbl == 0) & (od_lbl == 1)

            correct_idx_local = torch.where(correct_mask)[0].tolist()
            wrong_idx_local   = torch.where(wrong_mask)[0].tolist()

            correct_pairs.extend([(f_i, int(li)) for li in correct_idx_local])
            wrong_pairs.extend([(f_i, int(li)) for li in wrong_idx_local])

        return correct_pairs, wrong_pairs

    # ---- helper: collect activations/labels for sampled pairs ----
    def _collect_pairs(files: List[str], sampled_pairs: List[Tuple[int, int]]):
        by_file = defaultdict(list)
        for fi, local in sampled_pairs:
            by_file[fi].append(local)

        kept_ev_acts, kept_od_acts, kept_ev_lbls, kept_od_lbls = [], [], [], []
        for f_i, fpath in enumerate(files):
            locals_needed = by_file.get(f_i)
            if not locals_needed:
                continue
            data = torch.load(fpath, map_location="cpu")
            acts, labels = data["activations"], data["labels"]
            if acts.size(0) % 2 != 0:
                acts, labels = acts[:-1], labels[:-1]

            ev, od = acts[0::2], acts[1::2]
            ev_lbl, od_lbl = labels[0::2], labels[1::2]

            idx_tensor = torch.tensor(locals_needed, dtype=torch.long)
            kept_ev_acts.append(ev[idx_tensor])
            kept_od_acts.append(od[idx_tensor])
            kept_ev_lbls.append(ev_lbl[idx_tensor])
            kept_od_lbls.append(od_lbl[idx_tensor])

        sel_ev_acts = torch.cat(kept_ev_acts, dim=0)
        sel_od_acts = torch.cat(kept_od_acts, dim=0)
        sel_ev_lbls = torch.cat(kept_ev_lbls, dim=0)
        sel_od_lbls = torch.cat(kept_od_lbls, dim=0)

        return sel_ev_acts, sel_od_acts, sel_ev_lbls, sel_od_lbls

    # ================================================================
    # Main
    # ================================================================
    layers = _group_files_by_layer(input_dir)
    if 0 not in layers:
        raise ValueError("Layer 0 not found — cannot sample reference pairs.")

    if verbose:
        print(f"🔍 Found {len(layers)} layers with statements.")

    # Step 1: sample pairs from layer 0
    if verbose:
        print("\n📌 Sampling reference pairs from layer 0")
    correct_pairs, wrong_pairs = _scan_layer(layers[0])
    m = min(len(correct_pairs), len(wrong_pairs))
    sampled_correct = random.sample(correct_pairs, m)
    sampled_wrong   = random.sample(wrong_pairs, m)
    reference_pairs = sampled_correct + sampled_wrong
    random.shuffle(reference_pairs)
    if verbose:
        print(f"   ✅ Found {len(correct_pairs)} correct, {len(wrong_pairs)} wrong pairs in layer 0")
        print(f"   🎯 Chose {m} correct and {m} wrong pairs (total={2*m})")

    # ---- NEW: build dictionary of chosen stmts & indices ----
  # Save JSON of chosen pairs with original + final indices
    chosen_map = {}
    layer0_files = layers[0]
    final_idx_counter = 0
    for fi, local in reference_pairs:
        fname = os.path.basename(layer0_files[fi])
        if fname not in chosen_map:
            chosen_map[fname] = {"original_indices": [], "final_indices": []}
        chosen_map[fname]["original_indices"].append(local)
        # Each pair contributes 2 rows in final balanced tensor (even & odd)
        chosen_map[fname]["final_indices"].append(final_idx_counter)
        chosen_map[fname]["final_indices"].append(final_idx_counter + 1)
        final_idx_counter += 2
    
    chosen_json_path = os.path.join(output_dir, "chosen_pairs.json")
    with open(chosen_json_path, "w") as f:
        json.dump(chosen_map, f, indent=2)
    if verbose:
        print(f"   💾 Saved chosen pairs map with final indices -> {chosen_json_path}")

    summary = {}
    # Step 2: apply to all layers
    for layer_idx, files in tqdm(layers.items(), desc="Processing layers", unit="layer"):
        if verbose:
            print(f"\n📂 Layer {layer_idx}: {len(files)} stmt files")

        sel_ev_acts, sel_od_acts, sel_ev_lbls, sel_od_lbls = _collect_pairs(files, reference_pairs)
        
        # Pair up and shuffle
        pairs_acts = list(zip(sel_ev_acts, sel_od_acts))
        pairs_lbls = list(zip(sel_ev_lbls, sel_od_lbls))
        perm = torch.randperm(len(pairs_acts))
        shuf_acts, shuf_lbls = [], []
        for i in perm:
            ev_act, od_act = pairs_acts[i]
            ev_lbl, od_lbl = pairs_lbls[i]
            shuf_acts.append(ev_act)
            shuf_acts.append(od_act)
            shuf_lbls.append(ev_lbl)
            shuf_lbls.append(od_lbl)
        final_acts = torch.stack(shuf_acts, dim=0)
        final_lbls = torch.stack(shuf_lbls, dim=0)

        # Diagnostics
        ct_correct_TRUE  = int((sel_ev_lbls == 1).sum().item())
        ct_correct_FALSE = int((sel_od_lbls == 0).sum().item())
        ct_wrong_TRUE    = int((sel_ev_lbls == 0).sum().item())
        ct_wrong_FALSE   = int((sel_od_lbls == 1).sum().item())

        if verbose:
            print("   📊 Per-combo counts (before shuffle):")
            print(f"      correct+TRUE : {ct_correct_TRUE}")
            print(f"      correct+FALSE: {ct_correct_FALSE}")
            print(f"      wrong+TRUE   : {ct_wrong_TRUE}")
            print(f"      wrong+FALSE  : {ct_wrong_FALSE}")
            print(f"   Final rows after shuffle: {final_acts.size(0)}")

        out_path = os.path.join(output_dir, f"layer_{layer_idx}_balanced.pt")
        torch.save({"activations": final_acts, "labels": final_lbls}, out_path)

        summary[layer_idx] = {
            "saved": True,
            "pairs_per_class": m,
            "final_rows": int(final_acts.size(0)),
            "out_path": out_path,
        }
        if verbose:
            print(f"   💾 Saved -> {out_path}")

    if verbose:
        print("\n🎯 Done. All layers aligned to reference pairs from layer 0.")

    return summary
