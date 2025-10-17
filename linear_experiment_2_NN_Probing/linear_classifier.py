import torch
import os
import sys
import glob
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

layers = [0]

for l_idx in range(26):
    dataset_path = f"/home/current_run/activations/activations_svd_2304/activations_balanced/layer_{l_idx}_balanced_svd_processed.pt"
    output_path = f"/home/current_run/trained_probes"
    data = torch.load(dataset_path)
    
    # Convert to numpy
    X_data = data["activations"].numpy()
    y_data = data["labels"].numpy().ravel()
    
    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )
    
    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )
    print('hello1')
    # Create TensorDatasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    )
    
    print(f"Layer {l_idx}:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Train logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print("  Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))
    
    # Evaluate on test set
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"  Test Accuracy: {test_acc:.4f}")
    print("  Test Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Save linear classifier weights
    weights = torch.tensor(clf.coef_, dtype=torch.float32)
    bias = torch.tensor(clf.intercept_, dtype=torch.float32)
    
    model_dict = {
        "weights": weights,
        "bias": bias
    }
    print('hello')
    save_dir = os.path.join(output_path, "linear_classifier")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"linear_classifier_layer_{l_idx}.pt")
    torch.save(model_dict, save_path)
    
    print(f"Saved linear classifier weights to {save_path}\n")
