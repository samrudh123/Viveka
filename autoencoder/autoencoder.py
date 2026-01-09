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
    

class INNAutoencoderWithClassifier(nn.Module):
    def __init__(self, input_dim=2304, latent_dim=16,hidden_dim=256, num_coupling_layers=4):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim 
        self.z_dim = input_dim - latent_dim 

        # stacking coupling layers here

        self.coupling_layers = nn.ModuleList([
            CouplingLayer(input_dim, hidden_dim=hidden_dim)
            for _ in range(num_coupling_layers)
        ])


        # classification head on bottleneck layer

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )


    def forward(self,x,return_bottleneck=False):

        #forward through coupling layers
        output = x
        for layer in self.coupling_layers: 
            output = layer(output)

        # split into y and z along column dimension

        y = output[:, :self.latent_dim]
        z = output[:, self.latent_dim:]

        classification = self.classifier(y)

        # reconstruction with zero padding to force artificial bottleneck

        y_padded = t.cat([y, t.zeros_like(z)],dim=1)
        reconstructed = self.inverse(y_padded)

        if return_bottleneck:
            return reconstructed, classification, y,z
        return reconstructed, classification
    
    def inverse(self,y_z):
        #inverse pass through the INN

        output = y_z
        for layer in reversed(self.coupling_layers):
            output = layer.inverse(output)

        return output
    
    def encode(self,x):
        output = x
        for layer in self.coupling_layers:
            output = layer(output)

        y = output[:,:self.latent_dim]
        return y
    
    def classify(self,x):
        bottleneck = self.encode(x)
        return self.classifier(bottleneck)
    
def load_preprojected_dataset(projected_dir,layer_idx):

    activations_list, labels_list = [],[]
    file_pattern = os.path.join(projected_dir, f'layer_{layer_idx}_balanced.pt')

    for fname in tqdm(glob.glob(file_pattern), desc = f"loading layer number {layer_idx}", leave=False):
        data = t.load(fname, weights_only=False)
        activations_list.append(data['activations'])
        labels_list.append(data['labels'])

    if not activations_list:
        print("warning!! activations list was not made")
        return None
    
    return TensorDataset(
        t.cat(activations_list,dim=0),
        t.cat(labels_list,dim=0).float().unsqueeze(1)
    )

def train_inn_autoencoder_with_classifier(args):
    
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"using device : {device}")
    
    #HARDCODE ALERT!
    #change here if activation directories are different

    model_name_safe = args.model_name.replace('/','_')
    projected_dir = os.path.join(args.output_dir,"activations_balanced/")
    val_projected_dir = os.path.join(args.output_dir, "validation_activations/")
    autoencoders_dir = os.path.jpin(args.output_dir, "trained_inn_autoencoder_classifiers")

    os.makedirs(autoencoders_dir,exist_ok=True)

    print("train activations are from", projected_dir)
    print("validation activations are from", val_projected_dir)
    print(f"loss weights: alpha ={args.alpha} (classification) ; beta = {args.beta} (reconstruction), gamma = {args.gamma} (zero-padding)")

    #this outer for loop is for layerwise training

    for l_idx in tqdm(args.train_layers, desc = f"training autoencoder+classifier per layer at {autoencoders_dir}"):
        checkpoint_path = os.path.join(
            autoencoders_dir,
            f"inn_autoencoder_classifier_layer_{l_idx}_latent_{args.latent_dim}_alpha_{args.alpha}_beta_{args.beta}.pt"
        )

        early_stopper = EarlyStopping(patience=args.patience, save_path = checkpoint_path)

        if os.path.exists(projected_dir) and os.path.exists(val_projected_dir):
            train_files = glob.glob(os.path.join(projected_dir, f'layer_{l_idx}_balanced.pt'))
            val_files = glob.glob(os.path.join(val_projected_dir, f'layer_{l_idx}_balanced.pt'))

            if train_files and val_files:
                print(f"loading preprocessed train and validation data for layer {l_idx}")
                dataset_train = load_preprojected_dataset(projected_dir, l_idx)
                dataset_val = load_preprojected_dataset(val_projected_dir, l_idx)

            else :
                print(f"no data files for layer {l_idx}. Skipping")
                continue

        else : 
            print(f"no directories found for layer {l_idx}. skipping")
            continue

        if dataset_train is None or dataset_val is None:
            print(f"no data for layer {l_idx}. skipping")
            continue

        x_train, y_train = dataset_train.tensors
        x_val, y_val = dataset_val.tensors

        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        total_steps = len(train_loader) * args.num_epochs


        try:
            print(f"Attempting to load checkpoint from: {checkpoint_path}")
            
            checkpoint = t.load(checkpoint_path, map_location="cpu")
            
            model = INNAutoencoderWithClassifier(
                input_dim = args.input_dim,
                latent_dim = args.latent_dim,
                hidden_dim = args.hidden_dim,
                num_coupling_layers=args.num_coupling_layers
            ).to(device)

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer = t.optim.Adam(model.parameters(), lr=args.lr)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1

            print(f"\n--- resuming training for layer {l_idx} from epoch {start_epoch} ---")

        except Exception as e:
            print(f"No checkpoint found or error loading: {e}")

            model = INNAutoencoderWithClassifier(
                input_dim = args.input_dim,
                latent_dim = args.latent_dim,
                hidden_dim = args.hidden_dim,
                num_coupling_layers=args.num_coupling_layers
            ).to(device)

            optimizer = t.optim.Adam(model.parameters(), lr=args.lr)
            start_epoch = 0

            print(f"\n--- starting training from scratch for layer {l_idx} ---")

    
        reconstruction_criterion = nn.MSELoss() #is MSE valid here? can we use a better loss function since the dimensions are very high?? 
        classification_criterion = nn.BCELoss()
        zero_padding_criterion = nn.MSELoss() 

        def lr_lambda(current_step):
            if current_step < args.warmup_steps:
                return float(current_step) / float(max(1, args.warmup_steps))
            return 1.0
        
        scheduler = t.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
        writer = SummaryWriter(log_dir=os.path.join(args.logdir, f'layer_{l_idx}'))

        #wandb config

        config = {
            "layer" : l_idx,
            "learning_rate" : args.lr,
            "epochs" : args.num_epochs,
            "batch_size" : args.batch_size,
            "latent_dim" : args.latent_dim,
            "input_dim" : args.input_dim,
            "hidden_dim" : args.hidden_dim,
            "num_coupling_layers" : args.num_coupling_layers,
            "alpha" : args.alpha,
            "beta" : args.beta,
            "gamma" : args.gamma,
        }

        run_name = f"layer_{l_idx}_inn_ae_clf_latent{args.latent_dim}_alpha{args.alpha}_beta{args.beta}_lr{args.lr}"
        run = wandb.init(
            entity = args.wandb_entity,
            project = args.wandb_project,
            name = run_name,
            group = f"INN_epochs_{args.num_epochs}"
            config=config,
            reinit = True,
        )

        # training looop

        for epoch in range(start_epoch, args.num_epochs):

            model.train()
            epoch_recon_loss = 0.0
            epoch_class_loss = 0.0
            epoch_zero_loss = 0.0
            epoch_total_loss = 0.0
            train_labels = []
            train_preds = []
            start_time = time.time()


            #progress bar
            pbar = tqdm(train_loader, desc = f"Epoch {epoch+1}/{args.num_epochs} [Training]", leave=False)
            step = epoch * len(train_loader)

            for batch_idx, (X_batch, y_batch) in enumerate(pbar):
                try : 
                    #periodic saving
                    progress = batch_idx / len(train_loader)

                    if batch_idx % (len(train_loader) // 10 + 1) == 0 :
                        t.save ({
                            "model_state_dict" : model.state_dict(),
                            "optimizer_state_dict" : optimizer.state_dict(),
                            "epoch" : epoch
                        },checkpoint_path)
                        
                        print(f"model saved at {progress*100:.2f}% progress")

                    
                    X_batch = X_batch.to(t.float32).to(device)
                    y_batch = y_batch.to(t.float32).to(device)

                    optimizer.zero_grad()

                    reconstructed, classification, y_latent, z_latent = model(X_batch, return_botteneck=True)

                    if classification.shape != y_batch.shape:
                        print("warning!!! shape mismatch between y_batch and classification vector")
                        y_batch = y_batch.reshape(classification.shape)

                    
                    # how do we normalise the loss values? entropy would give lesser magnitude than MSE right? 
                    recon_loss = reconstruction_criterion(reconstructed, X_batch)
                    class_loss = classification_criterion(classification, y_batch)
                    zero_loss = zero_padding_criterion(z_latent, t.zeros_like(z_latent))

                    # total loss would be a weighted sum of the 3
                    total_loss = args.alpha * class_loss + args.beta * recon_loss + args.gamma * zero_loss

                    total_loss.backward()

                    t.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

                    optimizer.step()

                    if scheduler.get_last_lr()[0] < args.lr:
                        scheduler.step()

                    #metric logging 
                    current_recon_loss = recon_loss.item()
                    current_class_loss = class_loss.item()
                    current_zero_loss = zero_loss.item()
                    current_total_loss = total_loss.item()

                    epoch_recon_loss += current_recon_loss
                    epoch_class_loss += current_class_loss
                    epoch_zero_loss += current_zero_loss
                    epoch_total_loss += current_total_loss
                    
                    preds  = (classification > 0.5).float()
                    train_preds.extend(preds.cpu().numpy())
                    train_labels.extend(y_batch.cpu().numpy())

                    pbar.set_postfix({
                        'total' : current_total_loss,
                        'recon' : current_recon_loss,
                        'class' : current_class_loss,
                        'zero' : current_zero_loss,
                        'lr' : f"{scheduler.get_last_lr()[0]:.6f}"
                    })

                    batch_acc = accuracy_score(y_batch.cpu().numpy(),preds.cpu().numpy())
                    batch_f1 = f1_score(y_batch.cpu().numpy(), preds.cpu().numpy(), zero_division=0)

                    run.log({
                        "loss/train_total": current_total_loss,
                        "loss/train_reconstruction": current_recon_loss,
                        "loss/train_classification": current_class_loss,
                        "loss/train_zero_padding": current_zero_loss,
                        "acc/train": batch_acc,
                        "f1/train": batch_f1,
                        "learning_rate": scheduler.get_last_lr()[0]
                    })
                    log_memory(run)
                    step += 1

                except KeyboardInterrupt: 
                    t.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch
                    }, checkpoint_path)
                    print("Keyboard Interrupt, model saved")
                    run.finish()
                    return
                    

            
            avg_train_recon_loss = epoch_recon_loss / len(train_loader)
            avg_train_class_loss = epoch_class_loss / len(train_loader)
            avg_train_zero_loss = epoch_zero_loss / len(train_loader)
            avg_train_total_loss = epoch_total_loss / len(train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, zero_division=0)
            
            elapsed = time.time() - start_time
            est_remaining = (args.num_epochs - (epoch + 1)) * elapsed
            
            print(f"\nEpoch {epoch+1} completed in {elapsed:.2f}s | Estimated time remaining: {est_remaining:.2f}s")
            print(f"Train Total: {avg_train_total_loss:.6f} | Recon: {avg_train_recon_loss:.6f} | Class: {avg_train_class_loss:.6f} | Zero: {avg_train_zero_loss:.6f}")
            print(f"Train Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")
            
            writer.add_scalar("Loss/train_total", avg_train_total_loss, epoch)
            writer.add_scalar("Loss/train_reconstruction", avg_train_recon_loss, epoch)
            writer.add_scalar("Loss/train_classification", avg_train_class_loss, epoch)
            writer.add_scalar("Loss/train_zero_padding", avg_train_zero_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("F1/train", train_f1, epoch)
            
            run.log({
                "loss/train_total_epoch": avg_train_total_loss,
                "loss/train_reconstruction_epoch": avg_train_recon_loss,
                "loss/train_classification_epoch": avg_train_class_loss,
                "loss/train_zero_padding_epoch": avg_train_zero_loss,
                "acc/train_epoch": train_acc,
                "f1/train_epoch": train_f1
            })
            
            # Validation phase
            model.eval()
            val_recon_loss = 0.0
            val_class_loss = 0.0
            val_zero_loss = 0.0
            val_total_loss = 0.0
            val_preds = []
            val_labels = []
            
            with t.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Validation]", leave=False)
                
                for X_batch, y_batch in val_pbar:
                    X_batch = X_batch.to(t.float32).to(device)
                    y_batch = y_batch.to(t.float32).to(device)
                    
                    reconstructed, classification, y_latent, z_latent = model(X_batch, return_bottleneck=True)
                    
                    # Handle shape mismatch
                    if classification.shape != y_batch.shape:
                        y_batch = y_batch.reshape(classification.shape)
                    
                    recon_loss = reconstruction_criterion(reconstructed, X_batch)
                    class_loss = classification_criterion(classification, y_batch)
                    zero_loss = zero_padding_criterion(z_latent, t.zeros_like(z_latent))
                    total_loss = args.alpha * class_loss + args.beta * recon_loss + args.gamma * zero_loss
                    
                    val_recon_loss += recon_loss.item()
                    val_class_loss += class_loss.item()
                    val_zero_loss += zero_loss.item()
                    val_total_loss += total_loss.item()
                    
                    preds = (classification > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())
                    
                    val_pbar.set_postfix({'val_total_loss': total_loss.item()})
            
            avg_val_recon_loss = val_recon_loss / len(val_loader)
            avg_val_class_loss = val_class_loss / len(val_loader)
            avg_val_zero_loss = val_zero_loss / len(val_loader)
            avg_val_total_loss = val_total_loss / len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            
            print(f"Val Total: {avg_val_total_loss:.6f} | Recon: {avg_val_recon_loss:.6f} | Class: {avg_val_class_loss:.6f} | Zero: {avg_val_zero_loss:.6f}")
            print(f"Val Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
            print("Classification Report:\n", classification_report(val_labels, val_preds, digits=4, zero_division=0))
            
            writer.add_scalar("Loss/val_total", avg_val_total_loss, epoch)
            writer.add_scalar("Loss/val_reconstruction", avg_val_recon_loss, epoch)
            writer.add_scalar("Loss/val_classification", avg_val_class_loss, epoch)
            writer.add_scalar("Loss/val_zero_padding", avg_val_zero_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("F1/val", val_f1, epoch)
            
            run.log({
                "loss/val_total": avg_val_total_loss,
                "loss/val_reconstruction": avg_val_recon_loss,
                "loss/val_classification": avg_val_class_loss,
                "loss/val_zero_padding": avg_val_zero_loss,
                "acc/val": val_acc,
                "f1/val": val_f1
            })
            
            log_confusion_matrix(writer, val_labels, val_preds, epoch)
            
            # early stopping
            early_stopper(avg_val_total_loss, model, optimizer, epoch)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            #checkpoint
            t.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch
            }, checkpoint_path)
        
        writer.close()
        
        
        final_path = os.path.join(
            autoencoders_dir, 
            f'inn_autoencoder_classifier_layer_{l_idx}_latent_{args.latent_dim}_alpha_{args.alpha}_beta_{args.beta}_final.pt'
        )
        t.save(model.state_dict(), final_path)
        print(f"Final model saved to {final_path}")
        
        run.finish()



def main():
    parser = argparse.ArgumentParser(description='Train INN autoencoder with classification head on layer activations')
    
    # Model architecture parameters
    parser.add_argument('--input_dim', type=int, default=2304, help='Input dimension (d_model)')
    parser.add_argument('--latent_dim', type=int, default=16, help='Bottleneck/latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for coupling functions')
    parser.add_argument('--num_coupling_layers', type=int, default=4, help='Number of coupling layers')
    
    # Loss weighting parameters
    parser.add_argument('--alpha', type=float, default=0.5, 
                       help='Weight for classification loss')
    parser.add_argument('--beta', type=float, default=0.4, 
                       help='Weight for reconstruction loss')
    parser.add_argument('--gamma', type=float, default=0.1, 
                       help='Weight for zero-padding loss')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps for scheduler')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Data parameters
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory containing activations')
    parser.add_argument('--train_layers', type=int, nargs='+', required=True, help='Layers to train on')
    parser.add_argument('--model_name', type=str, default='gemma-2-2b-it', help='Model name')
    
    # Logging parameters
    parser.add_argument('--logdir', type=str, default='runs/inn_autoencoder_classifier', help='TensorBoard log directory')
    parser.add_argument('--wandb_entity', type=str, default='your-entity', help='WandB entity')
    parser.add_argument('--wandb_project', type=str, default='inn-autoencoder-classifier', help='WandB project name')
    
    args = parser.parse_args()
    
    # Create log directory
    os.makedirs(args.logdir, exist_ok=True)
    
    print("="*50)
    print("INN Autoencoder + Classifier Training Configuration")
    print("="*50)
    print(f"Input dim: {args.input_dim}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Coupling layers: {args.num_coupling_layers}")
    print(f"Loss weighting:")
    print(f"  - α (Classification): {args.alpha}")
    print(f"  - β (Reconstruction): {args.beta}")
    print(f"  - γ (Zero-padding): {args.gamma}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Layers: {args.train_layers}")
    print("="*50)
    
    train_inn_autoencoder_with_classifier(args)


if __name__ == '__main__':
    main()          

