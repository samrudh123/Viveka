from utils import load_model, load_statements
from hook import generate_and_label_answers, get_truth_probe_activations
from classifier import ProbingNetwork, hparams, log_confusion_matrix, lr_lambda
from svd_withgpu import perform_global_svd
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse
import glob
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, classification_report)
import time
import os
import torch as t
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
import random
import wandb
import psutil
from sklearn.model_selection import train_test_split

#SVD Projection Loader

def load_and_project_activations(activations_dir, layer_idx, device):
    """
    Loads raw activation files and the SVD projection matrix for a specific layer,
    then returns the projected data and labels.
    """
    svd_dir = os.path.join(os.path.dirname(activations_dir), 'svd_components')
    projection_matrix_path = os.path.join(svd_dir, f"projection_matrix_layer_{layer_idx}.pt")

    if not os.path.exists(projection_matrix_path):
        raise FileNotFoundError(
            f"SVD projection matrix not found for layer {layer_idx}. Please run the 'svd' stage first."
        )

    projection_matrix = t.load(projection_matrix_path).to(device)

    activations_list, labels_list = [], []
    file_pattern = os.path.join(activations_dir, f'layer_{layer_idx}_stmt_*.pt')

    for fname in tqdm(glob.glob(file_pattern), desc=f"Loading & Projecting L{layer_idx}", leave=False):
        data = t.load(fname)
        raw_activations = data['activations'].to(device)
        if raw_activations.dtype != projection_matrix.dtype:
            raw_activations = raw_activations.to(projection_matrix.dtype)
        projected_activations = (projection_matrix @ raw_activations.T).T
        activations_list.append(projected_activations)
        labels_list.append(data['labels'])

    if not activations_list:
        return None

    return TensorDataset(
        t.cat(activations_list, dim=0),
        t.cat(labels_list, dim=0).float().unsqueeze(1)
    )

#Training Functionality

def load_preprojected_dataset(projected_dir, layer_idx):
    """
    Loads already SVD-processed activation files for a specific layer
    from the activations_svd directory.
    """
    activations_list, labels_list = [], []
    #file_pattern = os.path.join(projected_dir, f'layer_{layer_idx}_balanced_svd_processed.pt')
    file_pattern = os.path.join(projected_dir, f'layer_{layer_idx}_balanced.pt')
    
    for fname in tqdm(glob.glob(file_pattern), desc=f"Loading pre-projected L{layer_idx}", leave=False):
        data = t.load(fname, weights_only=False)
        activations_list.append(data['activations'])
        labels_list.append(data['labels'])

    if not activations_list:
        return None

    return TensorDataset(
        t.cat(activations_list, dim=0),
        t.cat(labels_list, dim=0).float().unsqueeze(1)
    )


def train_probing_network(output_dir, train_layers, device):
    """
    Trains a probing network for each specified layer on either:
    - precomputed SVD-projected activations from activations_svd, or
    - raw activations projected on-the-fly using saved projection matrices.
    """
    
    #logging session metrics
    def get_ram_usage_gb():
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        return used_gb, total_gb

    def get_vram_usage_gb(device=0):
        used = t.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = t.cuda.memory_reserved(device) / (1024 ** 3)
        total = t.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        return used, reserved, total

    def log_memory(step=None):
        used_ram, total_ram = get_ram_usage_gb()
        used_vram, reserved_vram, total_vram = get_vram_usage_gb()

        run.log({
            "RAM_used_GB": used_ram,
            "VRAM_used_GB": used_vram,
            "VRAM_reserved_GB": reserved_vram,
        }, step=step)

    class EarlyStopping:
        global l_idx
        def __init__(self, patience=5, save_path = "checkpoint.pt", delta=0.0):
            """patience : be patient with violaiton for this many epochs
            save_path : where the model is stored if stopped
            delta : what threshold should be counted as a decrease in loss"""
            self.patience = patience
            self.save_path = save_path
            self.delta = delta
            self.counter=0
            self.best_score = None
            self.early_stop = False
        def __call__(self, val_loss, model, optimizer, epoch, hparams=None):
            score = -val_loss
            if self.best_score is None or score>self.best_score + self.delta:
                self.best_score = score
                self.counter=0
                t.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hparams": hparams
                }, self.save_path)
            else:
                self.counter+=1
                if self.counter>=self.patience:
                    self.early_stop=True

    model_name_safe = hparams.model_name.replace('/', '_')
    activations_dir = os.path.join(output_dir, 'activations', model_name_safe) ##
    #projected_dir = os.path.join(output_dir, 'activations/activations_svd_2304/activations_balanced')
    projected_dir = os.path.join(output_dir, 'activations/activations_balanced')
    probes_dir = os.path.join(output_dir, 'trained_probes', model_name_safe)
    os.makedirs(probes_dir, exist_ok=True)
    print(projected_dir, "Activations are being used from here")
    for l_idx in tqdm(train_layers, desc=f"Training probe per layer at {probes_dir}"):
        early_stopper = EarlyStopping(patience=5, save_path = f"/home/current_run/trained_probes/google_gemma-2-2b-it/2304_probe_model_layer_{l_idx}.pt") ###
        
        # Prefer precomputed projected activations if available
        #if os.path.exists(projected_dir) and glob.glob(os.path.join(projected_dir, f'layer_{l_idx}_balanced_svd_processed.pt')):
        if os.path.exists(projected_dir) and glob.glob(os.path.join(projected_dir, f'layer_{l_idx}_balanced.pt')):
            dataset = load_preprojected_dataset(projected_dir, l_idx)
            print(f"Found existing projections in {os.path.join(projected_dir, f'layer_{l_idx}_balanced.pt')}")
        else:
            dataset = load_and_project_activations(activations_dir, l_idx, device)

        if dataset is None:
            print(f"No data for layer {l_idx}. Skipping.")
            continue

        #train_size = int(0.8 * len(dataset))
        #val_size = len(dataset) - train_size
        x_data, y_data = dataset.tensors
        y_numpy = y_data.squeeze(1).cpu().numpy()
        X_train, X_test, y_train, y_test = train_test_split(x_data.cpu().numpy(), y_numpy, test_size=0.2, random_state=42, stratify=y_numpy)
        x_train_t = t.tensor(X_train, dtype=x_data.dtype)
        x_test_t  = t.tensor(X_test, dtype=x_data.dtype)
        y_train_t = t.tensor(y_train, dtype=y_data.dtype).unsqueeze(1)
        y_test_t  = t.tensor(y_test, dtype=y_data.dtype).unsqueeze(1)
        train_ds = TensorDataset(x_train_t, y_train_t)
        val_ds  = TensorDataset(x_test_t, y_test_t)
        train_loader = DataLoader(train_ds, batch_size=hparams.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=hparams.batch_size, shuffle=True)
        hparams.total_steps = len(train_loader) * hparams.num_epochs
        if os.path.exists(os.path.join(probes_dir, 'google_gemma-2-2b-it', f"2304_probe_model_layer_{l_idx}.pt")): ###
            checkpoint = t.load(f"/home/current_run/trained_probes/google_gemma-2-2b-it/2304_probe_model_layer_{l_idx}.pt", map_location="cpu") ###
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"]),
            start_epochs = checkpoint["epoch"]
            print(f"\n--- Training probe for Layer {l_idx} from epoch {start_epochs}. Existing train files found ---")
            step=0
            
        else:
            model = ProbingNetwork(hparams.model_name).to(device)
            optimizer = t.optim.Adam(model.parameters(), lr=hparams.lr)
            criterion = t.nn.BCELoss()
            scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            writer = SummaryWriter(log_dir=hparams.logdir)
            start_epochs = 0

            print(f"\n--- Training probe for Layer {l_idx} from start. No existing train files found ---")
            step = 0
        for epoch in range(hparams.num_epochs-start_epochs):
            config = {
            "layer":l_idx,
            "learning_rate": hparams.lr,
            "epochs": hparams.num_epochs,
            "probe_type": "trained-nn",
            "batch_size": hparams.batch_size
            }
            run_name = f"2304_layer_{config['layer']}_full_probes_{config['probe_type']}_lr{config['learning_rate']}_batchsize{config['batch_size']}"
            run = wandb.init(
                entity='jerrycloud3316-ai-club-iit-madras',
                project='nn on 2304 dim - probes for triviaqa on gemma-2-2b-it',
                name=run_name,
                group=f"epochs_{config['epochs']}",
                config=config,
            )
            epoch_loss = 0.0
            train_labels = []
            train_preds = []
            model.train()
            start_time = time.time()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+start_epochs+1}/{hparams.num_epochs} [Training]", leave=False)
            for X_batch, y_batch in pbar:
                try:
                    progress = pbar.n/pbar.total
                    if progress*100//10==0:
                        t.save({
                            "model_state_dict" : model.state_dict(),
                            "hparams" : hparams,
                            "optimizer_state_dict" : optimizer.state_dict(),
                            "epoch" : epoch
                        }, os.path.join(probes_dir, f'2304_probe_model_layer_{l_idx}.pt')) ###
                        print(f"model saved for this epoch at {progress*100}% progress")
                    X_batch, y_batch = X_batch.to(t.float32), y_batch.to(t.float32)
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    if scheduler.get_last_lr()[0] < hparams.lr:
                        scheduler.step()
                    pbar.set_postfix({'loss': loss.item(), 
                                      "lr": f"{scheduler.get_last_lr()[0]:.6f}"
                                      })
                    current_loss=loss.item()
                    epoch_loss += current_loss
                    preds = (outputs > 0.5).float()
                    train_preds.extend(preds.cpu().numpy())
                    train_labels.extend(y_batch.cpu().numpy())
                    step += 1
                    batch_acc = accuracy_score(y_batch.cpu().numpy(), preds.cpu().numpy())
                    batch_f1 = f1_score(y_batch.cpu().numpy(), preds.cpu().numpy())
                    run.log(
                        {
                            "loss/train":current_loss,
                            "acc/train":batch_acc,
                            "f1/train":batch_f1,
                            "learning-rate": scheduler.get_last_lr()[0]
                        }
                    )
                    log_memory()
                except KeyboardInterrupt:
                    t.save({
                        "model_state_dict" : model.state_dict(),
                        "hparams" : hparams,
                        "optimizer_state_dict" : optimizer.state_dict(),
                        "epoch" : epoch
                    }, os.path.join(probes_dir, f'2304_probe_model_layer_{l_idx}.pt')) ###
                    print("Keyboard Interrupt, model saved")
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds)
            avg_train_loss = epoch_loss / len(train_loader)
            elapsed = time.time() - start_time
            est_remaining = (hparams.num_epochs - (epoch + 1)) * elapsed

            print(f"\nEpoch {epoch+1} completed in {elapsed:.2f}s | Estimated time remaining: {est_remaining:.2f}s")
            print(f"Train Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")

            run.log(
                {
                    "loss/train":avg_train_loss,
                    "acc/train":train_acc,
                    "f1/train":train_f1,
                    "learning-rate": scheduler.get_last_lr()[0]
                }
            )
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("F1/train", train_f1, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            t.save({
                        "model_state_dict" : model.state_dict(),
                        "hparams" : hparams,
                        "optimizer_state_dict" : optimizer.state_dict(),
                        "epoch" : epoch
                    }, os.path.join(probes_dir, f'2304_probe_model_layer_{l_idx}.pt')) ###

            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            val_loss, correct, total = 0, 0, 0
            with t.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{hparams.num_epochs} [Validation]", leave=False)
                for X_batch, y_batch in val_pbar:
                    X_batch, y_batch = X_batch.to(t.float32), y_batch.to(t.float32)
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    current_loss=criterion(outputs, y_batch).item()
                    val_loss += current_loss
                    preds = (outputs > 0.5).float()
                    total += y_batch.size(0)
                    correct += (preds == y_batch).sum().item()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())
                    val_pbar.set_postfix({
                    "val_loss": f"{loss.item():.4f}"
                                        })
                    batch_acc = accuracy_score(y_batch.cpu().numpy(), preds.cpu().numpy())
                    batch_f1 = f1_score(y_batch.cpu().numpy(), preds.cpu().numpy()) 
                    run.log(
                    {
                        "loss/train":current_loss,
                        "acc/train":batch_acc,
                        "f1/train":batch_f1,
                        
                    } 
                    )  
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)
            avg_val_loss = val_loss / len(val_loader)
            early_stopper(avg_val_loss, model, optimizer, epoch, hparams)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
            print(f"Layer {l_idx} | Epoch {epoch+1} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {correct/total:.4f}")
            print(f"Val Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
            print("Classification Report:\n", classification_report(val_labels, val_preds, digits=4))
            

            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("F1/val", val_f1, epoch)
            
            run.log(
                {
                    "Loss/val": avg_val_loss,
                    "accuracy_val": val_acc,
                    "f1_val": val_f1
                }
            )
            log_confusion_matrix(writer, val_labels, val_preds, epoch)

        writer.close()
        run.finish()
        t.save(model.state_dict(), os.path.join(probes_dir, f'2304_probe_model_layer_{l_idx}.pt')) ###

#Main

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a multi-stage pipeline to generate data, extract activations, run SVD, and train a truth probe.")
    
    # --- Core Arguments ---
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument('--model_repo_id', type=str, required=True, default='google/gemma-2-2b-it', help="Hugging Face model repository ID.")
    parser.add_argument('--device', type=str, default='cuda' if t.cuda.is_available() else 'cpu')

    # --- Pipeline Stage Control ---
    parser.add_argument('--stage', type=str, choices=['generate', 'activate', 'svd', 'train', 'all'], default='all',
                        help="Which stage of the probing pipeline to run.")

    # --- Arguments for Parallelization ---
    parser.add_argument('--start_index', type=int, default=0, help="The starting row index of the dataset to process.")
    parser.add_argument('--end_index', type=int, default=None, help="The ending row index of the dataset to process(doesn't include this row). Processes to the end if not specified.")
    parser.add_argument('--gen_batch_size', type=int, default=1, help="Number of statements to process in parallel during generation.")
    #parser.add_argument('--batch_slice_arg', type=int, default=32, help="How many answers(*2) of each statement you wanna do at once.")
    
    # --- Generation Arguments ---
    parser.add_argument('--temperature', type=float, default=0.7, help="The temperature with which you want to generate completions, default=0.7")
    parser.add_argument('--top_p', type=float, default=0.9, help="Nucleus sampling threshold, default=0.9")
    parser.add_argument('--max_new_tokens', default=64, type=int, help="Max number of tokens to generate; default to 64. Specify -1 for no truncation")
    
    # --- Configuration Arguments ---
    parser.add_argument('--layers', nargs='+', type=int, default=[-1], help="List of layer indices to probe. -1 for all layers.")
    parser.add_argument('--probe_output_dir', type=str, default='current_run', help="Directory to save generated data and activations.")
    parser.add_argument('--num_generations', type=int, default=32, help="Number of answers to generate per statement for probing.")
    parser.add_argument('--generations_file', type=str, default='generated_completions_20k.json', help="Name of the JSON file with generated answers.")
    parser.add_argument('--generations_dir', type=str, default='generations', help="directory where the .json files are")

    # --- SVD & Training ---
    parser.add_argument('--svd_layers', nargs='+', type=int, help="Layers for 'svd' stage.")
    parser.add_argument('--train_layers', nargs='+', type=int, help="Layers for 'train' stage.")
    parser.add_argument('--svd_dim', type=int, default=576)

    args = parser.parse_args()
    hparams.model_name = args.model_repo_id
    print(args.max_new_tokens, "main_edited.py, argsparser")

    print(f"Loading model: {args.model_repo_id}...")

    df_unique = pd.read_json(r"./args.generations_dir/arg.generations_file")
    #df_unique
    row = df_unique.loc["generated_answers"]
    len_list = row.apply(len).tolist()
    avg = sum(len_list)/len(len_list)
    print(f"number of questions : {len(len_list)}\navg no. of unique ans per qn: {avg}")
 
    output_dir = args.probe_output_dir
    print(output_dir, "Output dir")
    if args.stage in ['generate', 'activate', 'all']:
        tokenizer, model, layer_modules = load_model(args.model_repo_id, args.device)
        if -1 in args.layers:
            args.layers = list(range(0,model.cfg.n_layers))
        num_layers = len(layer_modules)
        print(f"Loading dataset from: {args.dataset_path}")
        df, all_statements, all_correct_answers = load_statements(args.dataset_path)

        start = args.start_index
        end = args.end_index if args.end_index is not None else len(all_statements) #processes everything if not specified
        statements_to_process = all_statements[start:end]
        answers_to_process = all_correct_answers[start:end]

        if args.stage in ['generate', 'all']:
            batch_size = args.gen_batch_size
            for start_idx in range(0, len(statements_to_process), batch_size):
                end_idx = min(start_idx + batch_size, len(statements_to_process))
                batch_statements = statements_to_process[start_idx:end_idx]
                batch_answers = answers_to_process[start_idx:end_idx]

                generate_and_label_answers(
                    statements=batch_statements,
                    correct_answers=batch_answers,
                    tokenizer=tokenizer,
                    model=model,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                    num_generations=args.num_generations,
                    output_dir=args.probe_output_dir,
                    temperature=args.temperature,
                    top_p=args.top_p
                )

        if args.stage in ['activate', 'all']:
            batch_size = args.gen_batch_size
            for start_idx in tqdm(range(0, len(statements_to_process), batch_size), 
                total=len(statements_to_process)//batch_size + 1):
                end_idx = min(start_idx + batch_size, len(statements_to_process))
                batch_statements = statements_to_process[start_idx:end_idx]

                get_truth_probe_activations(
                    statements=batch_statements,
                    tokenizer=tokenizer,
                    model=model,
                    model_name_arg=args.model_repo_id.split('/')[-1],
                    #batch_size_arg=32,
                    layers=layer_modules,
                    layer_indices=args.layers,
                    device=args.device,
                    output_dir=args.probe_output_dir,
                    start_index=start_idx,
                    end_index=end_idx,
                    batch_list=len_list
                )

    if args.stage in ['svd', 'all']:
        if not args.svd_layers:
            parser.error("--svd_layers is required for 'svd' stage.")
        activations_dir = os.path.join(output_dir, 'activations/activations_balanced')
        perform_global_svd(activations_dir, args.svd_dim, args.svd_layers, args.device)
    


    if args.stage in ['train', 'all']:
        if not args.train_layers:
            parser.error("--train_layers is required for 'train' stage.")
        #acts_output = os.path.join(output_dir, "activations")
        train_probing_network(args.probe_output_dir, args.train_layers, args.device)
    if args.stage not in ['generate', 'activate', 'svd', 'train', 'all']:
        print("Invalid --stage arg")