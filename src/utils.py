import json
import os
import glob
import numpy as np
import torch
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_rmse_va(y_true, y_pred):
    squared_errors = np.sum((y_pred - y_true)**2, axis=1) # (Vp-Vg)^2 + (Ap-Ag)^2
    rmse_va = np.sqrt(np.mean(squared_errors))
    return rmse_va

def combine_training_datasets(root_folder, language='all'):

    combined_data = []

    # Verify root exists
    if not os.path.exists(root_folder):
        print(f"Error: Folder '{root_folder}' not found.")
        return []

    # Get all subfolders
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    print(f"Found {len(subfolders)} language folders.")

    for folder in subfolders:
        lang_code = os.path.basename(folder)

        # Filter by language if specific ones are requested
        if language != 'all' and lang_code != language:
            continue

        # Find all json files
        files = glob.glob(os.path.join(folder, "*.json*"))

        for file_path in files:
            filename = os.path.basename(file_path).lower()

            # Use only train data
            if 'train' in filename and 'dev' not in filename and 'test' not in filename:
                print(f"Loading: {filename} ({lang_code})")

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            content = json.load(f)
                            if isinstance(content, list):
                                combined_data.extend(content)
                        except json.JSONDecodeError:
                            f.seek(0)
                            for line in f:
                                if line.strip():
                                    combined_data.append(json.loads(line))
                except Exception as e:
                    print(f"Skipping {filename}: {e}")

    print(f"Combined Total: {len(combined_data)} samples")
    return combined_data

def ccc_loss(y_pred, y_true):
    loss = 0
    # Force as float for stability. Had issues with loss == nan
    y_pred = y_pred.float()
    y_true = y_true.float()
    eps = 1e-6
    
    if y_pred.size(0) < 2:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    for i in range(2):
        p, t = y_pred[:, i], y_true[:, i]
        mean_p, mean_t = torch.mean(p), torch.mean(t)
        # Unbiased=False is more stable for loss
        var_p = torch.var(p, unbiased=False)
        var_t = torch.var(t, unbiased=False)
        covariance = torch.mean((p - mean_p) * (t - mean_t))
        ccc = (2 * covariance) / (var_p + var_t + (mean_p - mean_t)**2 + eps)
        # Clamp for safety
        ccc = torch.clamp(ccc, min=-1.0, max=1.0)
        loss += (1 - ccc)
    return loss / 2

def calculate_ccc_numpy(y_true, y_pred):
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator