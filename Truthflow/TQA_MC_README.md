# TruthfulQA Multiple-Choice Dataset Support

This document describes the implementation of support for the TruthfulQA multiple-choice dataset in the TruthFlow framework.

## Overview

The TruthfulQA multiple-choice dataset has a different structure compared to the generation dataset. It contains `mc1_targets` and `mc2_targets` fields, each with `choices` and `labels`. This implementation converts it to a format compatible with the existing TruthFlow pipeline.

## Implementation Details

### 1. New Preprocessing Function

A new function `preprocess_tqa_mc` has been added to `utils.py` to process the multiple-choice dataset:

```python
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
```

This function extracts correct and incorrect answers from the `mc1_targets` field based on the labels (1 for correct, 0 for incorrect) and creates a new dataset with the same structure as the generation dataset.

### 2. Updated Dataset Loading

The `prepare_tqa_train_test_ds` function in `flow.py` has been updated to handle the multiple-choice dataset:

```python
def prepare_tqa_train_test_ds(tokenizer, ds_name, layers:List[int]=[13], is_mc=False):
    # ... existing code ...
    
    # If this is a multiple-choice dataset, we need to ensure it has the correct format
    if is_mc and 'mc1_targets' in test_ds.column_names:
        # Process the multiple-choice dataset to extract correct and incorrect answers
        test_ds = preprocess_tqa_mc(test_ds)
    
    # ... existing code ...
```

### 3. Updated Evaluation Functions

The evaluation functions (`flow_llm_mc`, `base_llm_mc`, and `base_llm`) have been updated to detect if the dataset is a multiple-choice dataset based on the dataset name and pass the appropriate flag to `prepare_tqa_train_test_ds`.

## Usage

To use the TruthfulQA multiple-choice dataset, follow these steps:

1. Create the dataset using `create_ds.py` with the `--ds_name tqa_mc` flag:

```bash
python create_ds.py --model_name gemma-2 --layers 20 --test_size 0.5 --seed 0 --token_pos ans_avg --ds_name tqa_mc --batch_size 1 --torch_dtype fp16
```

2. Run the evaluation using `flow.py` with the `--mc_eval` flag:

```bash
python flow.py --model_name gemma-2 --ds_path data_tqa/gemma-2_ans_avg_seed0_testsize0.5_layers_20 --layers 20 --seed 0 --truthflow --mc_eval --k 20 --alpha 1.5 --train --num_epochs 40
```

## Notes

- The multiple-choice dataset is processed to have the same structure as the generation dataset, so it can be used with the existing evaluation pipeline.
- The original `mc1_targets` and `mc2_targets` fields are preserved in the processed dataset for reference.
- The `is_mc` flag is determined automatically based on the dataset name (if it contains "tqa_mc").