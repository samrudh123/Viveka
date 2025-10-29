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
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

class hparams:
    """Hyperparameters for the 1D Autoencoder"""
    # Network dimensions
    input_dim = 2304      # <-- Set this to your 1D vector size
    hidden1 = 256
    hidden2 = 64
    latent_dim = 16     # This is the bottleneck dimension
    
    # Training parameters
    batch_size = 256
    lr = 1e-3
    num_epochs = 20
    warmup_steps = 100
    
    # Logging
    logdir = 'runs/autoencoder_1d'

hparams_1 = hparams()

# Create log directory if it doesn't exist
os.makedirs(hparams_1.logdir, exist_ok=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder part: compresses the input
        self.encoder = nn.Sequential(
            nn.Linear(hparams_1.input_dim, hparams_1.hidden1),
            nn.ReLU(),
            nn.Linear(hparams_1.hidden1, hparams_1.hidden2),
            nn.ReLU(),
            nn.Linear(hparams_1.hidden2, hparams_1.latent_dim)
        )
        
        # Decoder part: reconstructs from the latent space
        self.decoder = nn.Sequential(
            nn.Linear(hparams_1.latent_dim, hparams_1.hidden2),
            nn.ReLU(),
            nn.Linear(hparams_1.hidden2, hparams_1.hidden1),
            nn.ReLU(),
            nn.Linear(hparams_1.hidden1, hparams_1.input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def lr_lambda(current_step):
    if current_step < hparams_1.warmup_steps:
        return float(current_step) / float(max(1, hparams_1.warmup_steps))
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

# TODO : 
# 1. Add skip connections (make a U-net architechture)
# 2. Add a classification head on the bottleneck layer
# 3. Make 2 losses --> One classification loss, and one reconstruction loss
#    The goal of the algorithm would be to minimise this combined (optionally weighted) loss 

# we can try evaluating on other datasets --> this might give us better results than NN Probes