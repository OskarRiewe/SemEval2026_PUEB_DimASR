import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DebertaV2TokenizerFast, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from tqdm import tqdm

from model.GCN_deBERTA import DimABSA_GCN_Dataset, DebertaGCNRegressor
from model.deBERTA import DimABSA_Dataset, DebertaRegressor
from src.utils import ccc_loss, combine_training_datasets, set_seed, calculate_rmse_va

import transformers
transformers.utils.logging.set_verbosity_error()

def run_experiment(raw_data, lang, architecture='GCNdeBERTA'):
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> Starting Experiment | Lang: {lang.upper()} | Arch: {architecture} | Device: {device}")

    model_path = "microsoft/mdeberta-v3-base"
    loss_functions = ['MSE', 'HYBRID', 'CCC']
    n_folds     = 5
    epochs      = 5
    batch_size  = 64
    
    tokenizer   = DebertaV2TokenizerFast.from_pretrained(model_path)
    data_array  = np.array(raw_data, dtype=object)
    kf          = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Output Directory
    base_output_dir = os.path.join("results_v2", f"results_{lang}")
    os.makedirs(base_output_dir, exist_ok=True)

    for loss_type in loss_functions:
        for fold, (train_idx, val_idx) in enumerate(kf.split(data_array)):
            print(f"\n[Config: {architecture} | {loss_type} | Fold {fold+1}/{n_folds}]")

            train_raw = data_array[train_idx].tolist()
            val_raw = data_array[val_idx].tolist()

            if architecture == 'GCNdeBERTA':
                train_ds = DimABSA_GCN_Dataset(train_raw, tokenizer, lang)
                val_ds = DimABSA_GCN_Dataset(val_raw, tokenizer, lang)
                model = DebertaGCNRegressor(model_path).to(device)
                optimizer = torch.optim.AdamW([
                    {'params': model.deberta.parameters(), 'lr': 1e-5},
                    {'params': model.gcns.parameters(), 'lr': 2e-5},
                    {'params': model.regression_head.parameters(), 'lr': 2e-5},
                    {'params': model.layer_norm.parameters(), 'lr': 2e-5}
                ])
            else: # deBERTA Baseline
                train_ds = DimABSA_Dataset(train_raw, tokenizer)
                val_ds = DimABSA_Dataset(val_raw, tokenizer)
                model = DebertaRegressor(model_path).to(device)
                optimizer = torch.optim.AdamW([
                    {'params': model.deberta.parameters(), 'lr': 1e-5},
                    {'params': model.regression_head.parameters(), 'lr': 2e-5},
                    {'params': model.layer_norm.parameters(), 'lr': 2e-5}
                ])

            model           = model.float() # Full precision for stability
            train_loader    = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader      = DataLoader(val_ds, batch_size=batch_size)

            total_steps = len(train_loader) * epochs
            scheduler   = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

            # --- Training Loop ---
            for epoch in range(epochs):
                model.train()
                loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
                for batch in loop:
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    if architecture == 'GCNdeBERTA':
                        preds = model(input_ids, attention_mask, batch['adj_matrix'].to(device))
                    else:
                        preds = model(input_ids, attention_mask)

                    if loss_type == 'MSE':
                        loss = nn.MSELoss()(preds, labels)
                    elif loss_type == 'CCC':
                        loss = ccc_loss(preds, labels)
                    else:  # HYBRID
                        loss = nn.MSELoss()(preds, labels) + ccc_loss(preds, labels)
                    
                    if torch.isnan(loss):
                        print(f"Skipping batch: NaN detected in {loss_type} loss.")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    scheduler.step()
                    loop.set_postfix(loss=loss.item())

            # --- Evaluation ---
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    if architecture == 'GCNdeBERTA':
                        preds = model(input_ids, attention_mask, batch['adj_matrix'].to(device))
                    else:
                        preds = model(input_ids, attention_mask)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(batch['labels'].numpy())

            all_preds   = np.nan_to_num(np.vstack(all_preds))
            all_labels = np.vstack(all_labels)
            rmse_va = calculate_rmse_va(all_labels, all_preds)
            print(f"Fold {fold+1} Finished. RMSE: {rmse_va:.4f}")

            # --- Save Results ---
            filename = f"preds_{architecture}_{loss_type}_fold{fold+1}.csv"
            pd.DataFrame({
                'v_true': all_labels[:, 0], 'v_pred': all_preds[:, 0],
                'a_true': all_labels[:, 1], 'a_pred': all_preds[:, 1]
            }).to_csv(os.path.join(base_output_dir, filename), index=False)

if __name__ == "__main__":
    path = sys.argv[1]
    lang = sys.argv[2]
    arch_arg = sys.argv[3] if len(sys.argv) > 3 else 'all'
    
    all_data = combine_training_datasets(path, language=lang)
    if not all_data:
        print(f"No data found for language: {lang}")
        sys.exit(1)
        
    if arch_arg == 'all':
        archs_to_run = ['deBERTA', 'GCNdeBERTA']
    else:
        archs_to_run = [arch_arg]
        
    for current_arch in archs_to_run:
        run_experiment(all_data, lang=lang, architecture=current_arch)

    # Usage: python experiment_pipeline.py <data_path> <lang> [architecture or 'all']

