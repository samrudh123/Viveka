# TruthFlow: Truthful LLM Generation via Representation Flow Correction

This repository contains the source code and instructions to reproduce the results in our paper, **[TruthFlow: Truthful LLM Generation via Representation Flow Correction](https://arxiv.org/pdf/2502.04556)**.

All the errors have been rectified!

## Setup
Run the TruthFlow_Intervention.ipynb file (this file is made so that you can run it easily on colab)

You can use the required commands (for dataset creation, or flow model training) and comment out the other commands in the cell

The below sections are for explaining the commands being used in TruthFlow_Intervention.ipynb

## Create Dataset For TruthFlow
To train and test TruthFlow, you have to first extract query last token hidden states and query-specific truthful direction.
Here is an example command to create dataset.
```bash
python create_ds.py --model_name gemma-2 --layers 18 20 22 --test_size 0.5 --seed 0 --token_pos ans_avg --ds_name tqa
```
### Explanation of the command
* ``--model_name`` Specifies the model. Refer utils.py for model names supported
* ``--layers`` Which layer(s) to extract hidden states.
* ``--test_size`` How to split dataset.
* ``--seed`` Set random seed to ensure of reproducibility.
* ``--token_pos`` How to average hidden states for truthful direction.
* ``--ds_name`` What dataset to use.

## Train and Test TruthFlow

After collecting training data, the TruthFlow is ready to train and test. The following command will run the training and evaluation process.
```bash
python flow.py --model_name gemma-2 --ds_path data_tqa/gemma-2_ans_avg_seed0_testsize0.5_layers_18_20_22 --layers 20 --seed 0 --truthflow --mc_eval --k 20 --alpha 1.5 --train --num_epochs 40
```
### Explanation of the command
* ``--model_name`` Specifies the model.
* ``--layers`` Which layer to apply flow matching model. Should be only one layer!
* ``--seed`` Set random seed to ensure of reproducibility.
* ``--ds_path`` Local path to the data collected before for training and testing TruthFlow.
* ``--k`` How many top singular vectors to select to form the truthful subspace.
* ``--alpha`` The hyperparameter to control the intervention intensity. 
* ``--num_epochs`` How many epochs to train flow matching model. Use only when u mention ``--train`` otherwise ignore.
* Currently ``--open_gen_eval`` pipeline is not working so use ``--mc_eval`` for now.
* Use ``--truthflow`` for evaluating truthflow intervened model responses or ``-base`` for evaluating unintervened model responses.
* Mention ``--train`` if u want to train flow model. For evaluation ignore.

## Acknowledgements

This work builds upon several open source projects. In particular:

- The implementation of our rectified flow model follows the construction design from **[rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch) by Phil Wang (lucidrains)**. We are grateful to the authors for making their excellent implementation publicly available.


## How to cite
```bibtex
@misc{wang2025truthflowtruthfulllmgeneration,
      title={TruthFlow: Truthful LLM Generation via Representation Flow Correction}, 
      author={Hanyu Wang and Bochuan Cao and Yuanpu Cao and Jinghui Chen},
      year={2025},
      eprint={2502.04556},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.04556}, 
}

```

# TruthfulQA Multiple-Choice Dataset Support

This document describes the implementation of support for the TruthfulQA multiple-choice dataset in the TruthFlow framework.

## Overview

The TruthfulQA multiple-choice dataset has a different structure compared to the generation dataset. It contains `mc1_targets` and `mc2_targets` fields, each with `choices` and `labels`. This implementation converts it to a format compatible with the existing TruthFlow pipeline.

## Usage

To use the TruthfulQA multiple-choice dataset, follow these steps:

1. Create the dataset using `create_ds.py` with the `--ds_name tqa_mc` flag:

```bash
python create_ds.py --model_name gemma-2 --layers 20 --test_size 0.5 --seed 0 --token_pos ans_avg --ds_name tqa_mc --batch_size 1 --torch_dtype fp16
```

2. Run the evaluation using `flow.py` with the `--mc_eval` flag:

```bash
python flow.py --model_name gemma-2 --ds_path data_tqa_mc/gemma-2_ans_avg_seed0_testsize0.5_layers_20 --layers 20 --seed 0 --truthflow --mc_eval --k 20 --alpha 1.5 --train --num_epochs 40
```
