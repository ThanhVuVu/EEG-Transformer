import os
import glob
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 4 Main Clinical Classes for Chapman Dataset mapped to integers
SNOMED_MAPPING = {
    "426783006": 0, # Normal Sinus Rhythm (SR)
    "164889003": 1, # Atrial Fibrillation (AFIB)
    "426177001": 2, # Sinus Bradycardia (SB)
    "427084000": 3  # Sinus Tachycardia (ST)
}

class ChapmanDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        """
        Returns one ECG example for 4-class multi-class classification.

        Input type (per record):
            - WFDB/Chapman .mat contains a 12-lead waveform stored under `val`
            - Raw shape: (12, 5000)

        Output type (per __getitem__):
            - X tensor: shape (1, 12, 500), dtype float32
            - y tensor: scalar class id in {0,1,2,3}, dtype long
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Lazy Loading the .mat file into RAM (120KB per file)
        mat_data = loadmat(path)
        # WFDB matrices universally store the sensor graph inside the 'val' dictionary key
        signal = mat_data['val'] # Default shape: (12, 5000)
        
        # Sub-sampling strided sequence length to fit PyTorch max bounds efficiently
        # Compressing from 5000 samples -> 500 samples (preserve major waveform shapes)
        signal = signal[:, ::10] 
        
        # Pushing into Conv2D processing layout matching model limits: (Channels_spatial, H, W)
        # Model processes exactly (1 Depth map, 12 Sensors, 500 Subsequence Length)
        signal = np.expand_dims(signal, axis=0) # Shape -> (1, 12, 500)
        
        tensor_x = torch.tensor(signal, dtype=torch.float32)
        tensor_y = torch.tensor(label, dtype=torch.long)
        
        return tensor_x, tensor_y

def parse_hea_label(hea_path):
    """Parses SNOMED strings out of the WFDB header bounds."""
    with open(hea_path, 'r') as f:
        for line in f:
            if line.startswith("#Dx:"):
                dx_str = line.replace("#Dx:", "").strip()
                codes = [code.strip() for code in dx_str.split(",")]
                # Map onto our filtered classes, using first diagnosis match found
                for code in codes:
                    if code in SNOMED_MAPPING:
                        return SNOMED_MAPPING[code]
    return None

def prepare_data_lists(data_dir):
    """Locates all records and restricts matching ones to 4 classes."""
    all_hea = glob.glob(os.path.join(data_dir, "*.hea"))
    valid_paths = []
    valid_labels = []
    
    for hea in all_hea:
        label = parse_hea_label(hea)
        if label is not None:
            mat_path = hea.replace(".hea", ".mat")
            if os.path.exists(mat_path):
                valid_paths.append(mat_path)
                valid_labels.append(label)
                
    return valid_paths, valid_labels

def balance_indices(labels, rng=None):
    """Enforces class-balance matching the median frequency class distribution limits."""
    if rng is None:
        rng = np.random.default_rng(42)

    unique_classes, counts = np.unique(labels, return_counts=True)
    target_count = int(np.median(counts))
    print(f"Balancing Train dataset: Targeting uniform {target_count} samples per diagnosis class.")
    
    balanced_idx = []
    labels = np.array(labels)
    
    for cls in unique_classes:
        idx = np.where(labels == cls)[0]
        if len(idx) >= target_count:
            # Undersample the dominant class
            chosen = rng.choice(idx, target_count, replace=False)
        else:
            # Oversample the minority class
            repeats = target_count // len(idx)
            rem = target_count % len(idx)
            if rem > 0:
                chosen = np.concatenate([np.repeat(idx, repeats), rng.choice(idx, rem, replace=False)])
            else:
                chosen = np.repeat(idx, repeats)
        balanced_idx.extend(chosen)
    
    balanced_idx = np.array(balanced_idx)
    rng.shuffle(balanced_idx)
    return balanced_idx.tolist()

def get_dataloaders(data_dir, batch_size=64, random_state=42):
    """Core function exporting the Pipeline Loaders to the Model."""
    print(f"Scanning raw root tree {data_dir} ...")
    paths, labels = prepare_data_lists(data_dir)
    print(f"Successfully processed {len(paths)} clinical matrices matching Target Maps.")
    
    # 80 / 10 / 10 Deterministic Split System
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        paths,
        labels,
        test_size=0.1,
        stratify=labels,
        random_state=random_state
    )
    # The remaining 90% is split 8/9 Train (~80% total) and 1/9 Val (10% total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=1/9,
        stratify=y_train_val,
        random_state=random_state
    )
    
    print(f"Split Volumes -> Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Only the TRAIN split undergoes balancing; Val/Test remain as stratified (real-world) frequencies.
    # This preserves a clean evaluation protocol.
    rng = np.random.default_rng(random_state)
    bal_train_idx = balance_indices(y_train, rng=rng)
    X_train_bal = [X_train[i] for i in bal_train_idx]
    y_train_bal = [y_train[i] for i in bal_train_idx]
    
    # Export bounds into Dataset formats dynamically loading on call requests
    train_ds = ChapmanDataset(X_train_bal, y_train_bal)
    val_ds = ChapmanDataset(X_val, y_val)
    test_ds = ChapmanDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    dl, vl, tl = get_dataloaders(r"d:\12 lead ECG\WFDB_ChapmanShaoxing", batch_size=64)
    x, y = next(iter(dl))
    print(f"Tensor Pipeline Verified. Exported exact boundaries: Tensor Shape {x.shape}, Diagnostic Nodes {y.shape}")
