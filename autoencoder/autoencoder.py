import torch as t
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import numpy as np
import os
import glob
import argparse
import psutil 
import wandb

def get_ram_usage_gb():
    mem = psutil.virtual_memeory()
    used_gb = mem.used / (1024**3)
    total_gb =  mem.total / (1024**3)
    return used_gb, total_gb

def get_vram_usage_gb(device=0):
    used = t.cuda.memory_allocated(device) / (1024**3)
    reserved = t.cuda.memory_reserved(device) / (1024**3)
    total = t.cuda.get_device_properties(device).total_memory / (1024**3)

    return used, reserved, total

def log_memory(run, step = None):
    used_ram, total_ram = get_ram_usage_gb()
    used_vram, reserved_vram, total_vram = get_vram_usage_gb()

    run.log({
        "RAM_used_GB" : used_ram,
        "VRAM_used_GB" : used_vram,
        "VRAM_reserved_GB" : reserved_vram,
    }, step = step)

def log_confusion_matrix(writer,labels,preds,epoch,class_names=['0','1']):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True,fmt="d",cmap="Blues",xticklabels=class_names,yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    writer.add_figure("ConfusionMatrix/val", fig, global_step=epoch)
    plt.close(fig)


class EarlyStopping:
    def __init__(self,patience=5,save_path="checkpoint.pt",delta=0.0):
        self.patience = patience
        self.save_path = save_path
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss,model,optimizer,epoch,hparams=None):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            t.save ({
                "epoch" : epoch,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "hparams" : hparams
            }, self.save_path)

        else : 
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# here I implement the coupling layer from the INN paper as it is

class CouplingLayer(nn.Module):

    def __init__(self,dim,hidden_dim=256,clamp=2.0):
        super().__init__
        self.clamp = clamp

        # first split

        self.s1 = nn.Sequential(
            nn.Linear(dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2)
        )

        self.t1 = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2)
        )
        
        # second split
        self.s2 = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2)
        )
        
        self.t2 = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2)
        )

    def forward(self ,x) : 
        x1, x2 = t.chunk(x, 2, dim=1)


        # clamp is there to control the model output so that it does not explode/vanish since we are 
        # gonna use exponentials soon
        s2_out = self.s2(x2).clamp(-self.clamp, self.clamp)
        y1 = x1 * t.exp(s2_out) + self.t2(x2)

        s1_out = self.s1(y1).clamp(-self.clamp,self.clamp)
        y2 = x2 * t.exp(s1_out) + self.t1(y1)

        y = t.cat([y1,y2],dim=1)
        return y
    
    def inverse(self,y):
        y1, y2 = t.chunk(y, 2, dim=1)

        #first inverse transformation
        s1_out = self.s1(y1).clamp(-self.clamp,self.clamp)
        x2 = (y2 - self.t1(y1)) * t.exp(-s1_out)

        # second inverse transformation

        s2_out = self.s2(x2).clamp(-self.clamp,self.clamp)
        x1 = (y1 - self.t2(x2)) * t.exp(-s2_out)

        x = t.cat([x1,x2],dim=1)
        return x
    


