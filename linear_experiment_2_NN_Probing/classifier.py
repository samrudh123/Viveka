# classifier.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score, 
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import io
from tqdm import tqdm
import time
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


class HParams:
    input_dim = 576
    hidden1 = 144
    hidden2 = 12
    output_dim = 1
    batch_size = 32
    lr = 1e-4
    num_epochs = 3
    warmup_steps = 100
    model_name = 'gemma-2-2b-it'
    logdir = 'runs/synth'


hparams = HParams()


class ProbingNetwork(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.net = nn.Sequential(
            nn.Linear(hparams.input_dim, hparams.hidden1),
            nn.ReLU(),
            nn.Linear(hparams.hidden1, hparams.hidden2),
            nn.ReLU(),
            nn.Linear(hparams.hidden2, hparams.output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        #print(f'Using model: {self.model_name}')
        return self.net(x)


def lr_lambda(current_step):
    if current_step < hparams.warmup_steps:
        return float(current_step) / float(max(1, hparams.warmup_steps))
    return 1.0


def log_confusion_matrix(writer, labels, preds, epoch, class_names=['0', '1']):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    writer.add_figure("ConfusionMatrix/val", fig, global_step=epoch)
    plt.close(fig)