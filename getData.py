import os
import glob
import pandas as pd
import numpy as np
import torch

def get_aami_mapping():
    """
    Returns the AAMI EC57 standard class mapping for heartbeat types.
    Groups MIT-BIH annotations into 5 main classes:
    0: N (Normal)
    1: S (Supraventricular ectopic beat)
    2: V (Ventricular ectopic beat)
    3: F (Fusion)
    4: Q (Unknown/Unclassifiable)
    """
    return {
        'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, 
        'A': 1, 'a': 1, 'J': 1, 'S': 1,         
        'V': 2, 'E': 2,                         
        'F': 3,                                 
        '/': 4, 'f': 4, 'Q': 4                  
    }

def parse_annotations(file_path):
    """ Reads the annotations.txt file, returning peaks and labels arrays. """
    peaks, labels = [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:] # skip header
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            try:
                peaks.append(int(parts[1]))
                labels.append(parts[2])
            except ValueError:
                continue
    return np.array(peaks), np.array(labels)

def balance_dataset(X, y, target_samples_per_class=None):
    """
    Balances class distributions.
    Majority classes are randomly undersampled (without replacement).
    Minority classes are randomly oversampled (with replacement).
    """
    unique_classes, counts = torch.unique(y, return_counts=True)
    
    if target_samples_per_class is None:
        # Default strategy: balance to the median frequency to prevent huge oversampling or dropping too much
        target_samples_per_class = int(np.median(counts.numpy()))
        print(f"\nAuto-Balance active: Targeting precisely {target_samples_per_class} instances per class.")

    balanced_X = []
    balanced_y = []

    for cls in unique_classes:
        indices = torch.where(y == cls)[0]
        n_available = len(indices)
        
        if n_available >= target_samples_per_class:
            # Undersample majority class
            chosen_idx = indices[torch.randperm(n_available)[:target_samples_per_class]]
        else:
            # Oversample minority class
            repeats = target_samples_per_class // n_available
            rem = target_samples_per_class % n_available
            rem_idx = indices[torch.randperm(n_available)[:rem]]
            chosen_idx = torch.cat([indices.repeat(repeats), rem_idx])
            
        # Shuffle class indices to prevent block sequences if needed
        chosen_idx = chosen_idx[torch.randperm(len(chosen_idx))]    
            
        balanced_X.append(X[chosen_idx])
        balanced_y.append(y[chosen_idx])
        
    X_bal = torch.cat(balanced_X)
    y_bal = torch.cat(balanced_y)
    
    # Global shuffle to mix classes evenly
    perm = torch.randperm(len(y_bal))
    return X_bal[perm], y_bal[perm]

def load_mitbih_data(data_dir, window_size=250, balance=True, target_samples=None):
    """
    Loads MIT-BIH CSV recordings & annotations. Segments by 250 sample windows around R-peaks.
    Allows for automatic uniform data balancing to avoid dominant class 0.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    all_data = []
    all_labels = []
    
    aami_map = get_aami_mapping()
    half_window = window_size // 2
    right_window = window_size - half_window
    
    print(f"Found {len(csv_files)} records... Gathering raw signal windows")
    
    for csv_file in csv_files:
        base_name = os.path.basename(csv_file).replace('.csv', '')
        annot_file = os.path.join(data_dir, f"{base_name}annotations.txt")
        
        if not os.path.exists(annot_file): continue
            
        try:
            df = pd.read_csv(csv_file)
            signals = df.iloc[:, 1:3].values 
        except Exception:
            continue
            
        peaks, labels = parse_annotations(annot_file)
        
        for peak, label in zip(peaks, labels):
            if label not in aami_map: continue
                
            y_cls = aami_map[label]
            start_idx, end_idx = peak - half_window, peak + right_window
            
            if start_idx < 0 or end_idx > len(signals): continue
                
            window_data = signals[start_idx:end_idx].T 
            all_data.append(window_data)
            all_labels.append(y_cls)
            
    if not all_data:
        raise ValueError("No valid beats found.")
        
    # Standard output formatting (Batch, 1, Channels, Sequence)
    X = np.expand_dims(np.stack(all_data), axis=1)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(np.array(all_labels), dtype=torch.long)
    
    print(f"Raw shape before balancing: {X_tensor.shape}")
    
    if balance:
        X_tensor, y_tensor = balance_dataset(X_tensor, y_tensor, target_samples)
        
    print(f"\nFinal Features: {X_tensor.shape} \nFinal Labels: {y_tensor.shape}")
    return X_tensor, y_tensor

def get_dataloaders(data_dir, batch_size=64, train_ratio=0.8, balance=True):
    from torch.utils.data import TensorDataset, DataLoader, random_split
    
    X, y = load_mitbih_data(data_dir, balance=balance)
    dataset = TensorDataset(X, y)
    
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == '__main__':
    data_folder = r"d:\archive"
    # By default, load_mitbih_data now balances using the median counts
    X, y = load_mitbih_data(data_folder, window_size=250, balance=True)
    
    unique_labels, counts = np.unique(y.numpy(), return_counts=True)
    print("\nBalanced Class Distribution (AAMI):")
    class_names = ['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)', 'F (Fusion)', 'Q (Unknown)']
    for lbl, count in zip(unique_labels, counts):
        if lbl < len(class_names):
            print(f"  Class {lbl} - {class_names[lbl]}: {count} beats")
