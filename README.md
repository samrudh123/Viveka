# Viveka: Mitigating Hallucinations using Mechanistic Interpretability

The code base supports the following experiments:
- Probing Large Language Models
- Toy Transformers
- Recreating the paper 'LLMs know more than they Show'
- Reimplementing 'Tuned Lens'
- Basic utilities for circuit identification and activation visualization

## Probing Large Language Models
### Overview
This repository investigates whether large language models (LLMs) internally represent factual truth in a way that can be decoded by simple probes. The core, production-ready pipeline lives in `linear_experiment_2_NN_Probing/` and implements a modular workflow to:
- generate model completions
- extract internal activations
- perform dimensionality reduction via SVD
- train a small neural network probe to classify truthfulness

The pipeline is designed to scale to large datasets with batching and caching, and to run each stage independently or end-to-end.

### Key Capabilities
- Modular stages: `generate`, `activate`, `svd`, `train`, or `all`
- Cached generation to avoid recompute
- Counterfactual inputs ("… True" / "… False") to balance labels
- Per-layer activation capture via forward hooks
- Global SVD and on-the-fly or precomputed projection
- Simple MLP probe with TensorBoard logging

---

## Repo Structure (high-level)
- `linear_experiment_2_NN_Probing/`
  - `main_edited.py`: Entry point; orchestrates all stages and CLI.
  - `hook.py`: Generation and activation extraction.
  - `utils.py`: HF model/tokenizer loading and generation utilities.
  - `svd_withgpu.py`: Global SVD and per-statement projection writer.
  - `classifier.py`: Probe architecture and training helpers (metrics, logging).
- Other research directories (not the focus of this README):
  - `linear_experiments/`, `experiment_1/`, `toy_transformer/`, `truthful_behavior_universal/`, `lens/`, `circuit/`

---

## Pipeline Details (linear_experiment_2_NN_Probing)

### Data Flow
```
[Dataset CSV] → generate → [generations/<model>_generations.json]
             → activate → [activations/<model>/layer_{L}_stmt_{i}.pt]
             → svd      → [svd_components/projection_matrix_layer_{L}.pt]
                        → [activations_svd/<model>/layer{L}_stmt{i}_svd_processed.pt]
             → train    → [trained_probes/<model>/probe_model_layer_{L}.pt]
```

### Stage 1: Generate (`--stage generate`)
- Reads statements and ground-truth answers from the CSV.
- Applies a model-appropriate prompt template.
- Generates `--num_generations` answers per statement in batch.
- Labels each generation via fuzzy match against the truth list (threshold ~90).
- Caches results in `generations/<model_safe>_generations.json`.

### Stage 2: Activate (`--stage activate`)
- Loads cached generations.
- Builds counterfactual inputs per generation: "… True" and "… False".
- Assigns labels so each pair is balanced (true/false flipped by correctness).
- Registers forward hooks on selected transformer layers.
- Saves last-token residual activations per statement per layer to
  `activations/<model_safe>/layer_{L}_stmt_{global_idx}.pt` with tensors:
  - `activations`: shape `[2 * num_generations, d_model]`
  - `labels`: shape `[2 * num_generations]` in {0,1}

### Stage 3: SVD (`--stage svd`)
- For each layer, concatenates all raw activations across statements.
- Runs SVD (GPU-first, CPU fallback). Takes top `--svd_dim` right-singular vectors.
- Saves projection matrix to `svd_components/projection_matrix_layer_{L}.pt`.
- Projects each statement file and writes to
  `activations_svd/<model_safe>/layer{L}_stmt{i}_svd_processed.pt` (dtype preserved).

### Stage 4: Train (`--stage train`)
- Prefers precomputed SVD-projected files when available; otherwise projects on the fly using saved matrices.
- Splits into 80/20 train/val per-layer.
- Trains a small MLP (`classifier.ProbingNetwork`) on binary labels with BCE loss.
- Logs metrics and confusion matrices to TensorBoard (`runs/...`).
- Saves probe weights to `trained_probes/<model_safe>/probe_model_layer_{L}.pt`.

---

## Installation
Use a recent Python (3.10+) with CUDA if available.

```bash
pip install torch transformers thefuzz python-levenshtein scikit-learn pandas tqdm tensorboard
```

Note: The code sets `pad_token` for chat models if missing and uses BF16/FP16 when available.

---

## Command Line (main_edited.py)
Required inputs come from the CSV with columns:
- `statement` or `raw_question`
- `label` or `correct_answer` (list-like or string)

Common flags:
- `--dataset_path` (str, required): Path to CSV.
- `--model_repo_id` (str, required): HF model id, e.g. `google/gemma-2-2b-it`.
- `--device` (str): `cuda` or `cpu` (auto default).
- `--stage` {generate, activate, svd, train, all}
- `--start_index` / `--end_index`: Slice the dataset for batching/parallelism.
- `--gen_batch_size` (int): Statements processed per batch in `generate`.
- Generation: `--temperature` (0.7), `--top_p` (0.9), `--max_new_tokens` (64), `--num_generations` (32).
- Selection: `--layers` (e.g., `0 5 10`, or `-1` for all model layers).
- IO: `--probe_output_dir` (default `/kaggle/working/current_run`).
- SVD: `--svd_layers` (list), `--svd_dim` (default 576).
- Train: `--train_layers` (list).

---

## Example Runs

### 1) Generate (cached) and Activate for a slice
```bash
python linear_experiment_2_NN_Probing/main_edited.py \
  --dataset_path path/to/dataset.csv \
  --model_repo_id google/gemma-2-2b-it \
  --stage activate \
  --start_index 0 \
  --end_index 2000 \
  --gen_batch_size 4 \
  --num_generations 32 \
  --probe_output_dir current_run \
  --layers -1
```
Note: `activate` will use existing generations or call the generation logic for missing statements in the slice.

### 2) SVD on all layers to 576 dims
```bash
python linear_experiment_2_NN_Probing/main_edited.py \
  --dataset_path path/to/dataset.csv \
  --model_repo_id google/gemma-2-2b-it \
  --stage svd \
  --probe_output_dir current_run \
  --svd_layers -1 \
  --svd_dim 576
```

### 3) Train probes
```bash
python linear_experiment_2_NN_Probing/main_edited.py \
  --dataset_path path/to/dataset.csv \
  --model_repo_id google/gemma-2-2b-it \
  --stage train \
  --probe_output_dir current_run \
  --train_layers -1
```

### 4) End-to-end
```bash
python linear_experiment_2_NN_Probing/main_edited.py \
  --dataset_path path/to/dataset.csv \
  --model_repo_id google/gemma-2-2b-it \
  --stage all \
  --probe_output_dir current_run \
  --layers -1 \
  --svd_layers -1 \
  --train_layers -1 \
  --svd_dim 576
```

---

## Outputs
Within `--probe_output_dir` (e.g., `current_run/`):
```
current_run/
├─ generations/
│  └─ google_gemma-2-2b-it_generations.json
├─ activations/
│  └─ google_gemma-2-2b-it/
│     ├─ layer_0_stmt_0.pt
│     ├─ layer_0_stmt_1.pt
│     └─ ...
├─ svd_components/
│  ├─ projection_matrix_layer_0.pt
│  └─ ...
├─ activations_svd/
│  └─ google_gemma-2-2b-it/
│     ├─ layer0_stmt0_svd_processed.pt
│     └─ ...
└─ trained_probes/
   └─ google_gemma-2-2b-it/
      ├─ probe_model_layer_0.pt
      └─ ...
```

---

## Notes & Tips
- If `tokenizer.pad_token` is missing, it will be set automatically.
- For very large activation sets, SVD may fall back to CPU due to GPU memory limits.
- You can train without pre-saved `activations_svd/...` because `train` will project on the fly using `svd_components` if needed.
- TensorBoard logs are written under `runs/...` (see `classifier.py`).

---

## Citation
If you build on this codebase, please cite the repository and specify that you used the `linear_experiment_2_NN_Probing` probing pipeline.
