import os
import pandas as pd
import numpy as np
import torch

def balance_dataset(X, y, target_samples_per_class=None):
    """
    Balances class distributions.
    Majority classes are randomly undersampled. Minority classes oversampled with replacement.
    """
    unique_classes, counts = torch.unique(y, return_counts=True)
    
    if target_samples_per_class is None:
        target_samples_per_class = int(np.median(counts.numpy()))
        print(f"Auto-Balance active: Targeting precisely {target_samples_per_class} instances per class.")

    balanced_X = []
    balanced_y = []

    for cls in unique_classes:
        indices = torch.where(y == cls)[0]
        n_available = len(indices)
        
        if n_available >= target_samples_per_class:
            chosen_idx = indices[torch.randperm(n_available)[:target_samples_per_class]]
        else:
            repeats = target_samples_per_class // n_available
            rem = target_samples_per_class % n_available
            rem_idx = indices[torch.randperm(n_available)[:rem]]
            chosen_idx = torch.cat([indices.repeat(repeats), rem_idx])
            
        chosen_idx = chosen_idx[torch.randperm(len(chosen_idx))]    
            
        balanced_X.append(X[chosen_idx])
        balanced_y.append(y[chosen_idx])
        
    X_bal = torch.cat(balanced_X)
    y_bal = torch.cat(balanced_y)
    
    perm = torch.randperm(len(y_bal))
    return X_bal[perm], y_bal[perm]

def load_mitbih_kag_data(data_file, balance=True, target_samples=None):
    """
    Loads the preprocessed Kaggle Heartbeat Categorization Dataset.
    Columns 0-186 are sequence tensors. Column 187 is the integer label.
    """
    print(f"Extracting memory block from {data_file}...")
    df = pd.read_csv(data_file, header=None)
    
    # Slicing the dataframe
    features = df.iloc[:, :-1].values  # Shape: (N, 187)
    labels = df.iloc[:, -1].values      # Shape: (N,)
    
    # Expand to user-requested architecture parsing format: (batch, 1, 1, 187)
    X = np.expand_dims(features, axis=1) # (N, 1, 187)
    X = np.expand_dims(X, axis=1)        # (N, 1, 1, 187)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    
    print(f"Raw un-balanced shape: {X_tensor.shape}")
    
    if balance:
        X_tensor, y_tensor = balance_dataset(X_tensor, y_tensor, target_samples)
        
    print(f"Final Input Tensor: {X_tensor.shape} \nOutput Label Map: {y_tensor.shape}\n")
    return X_tensor, y_tensor

def get_dataloaders(data_dir, batch_size=64, balance=True, target_samples=None):
    from torch.utils.data import TensorDataset, DataLoader
    
    train_file = os.path.join(data_dir, "mitbih_train.csv")
    test_file = os.path.join(data_dir, "mitbih_test.csv")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Could not find {train_file}")
        
    # We heavily balance the training set to prevent bias, but we leave the TEST
    # set completely raw and unbounded so accuracy translates to real-world skew probability.
    X_train, y_train = load_mitbih_kag_data(train_file, balance=balance, target_samples=target_samples)
    X_test, y_test = load_mitbih_kag_data(test_file, balance=False)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == '__main__':
    data_folder = r"d:\archive (1)"
    get_dataloaders(data_folder, balance=True)
    print("Testing functionality passed correctly!")
