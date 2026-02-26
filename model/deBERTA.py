import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Model
import torch
import pandas as pd

class DimABSA_Dataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=128):
        self.encodings = {
            "input_ids": [],
            "attention_mask": []
        }
        self.labels = []

        for data in data_list:
            text = data.get('Text', '')
            items = data.get('Quadruplet') or data.get('Aspect_VA')

            if not items:
                continue

            for item in items:
                raw_aspect = item.get('Aspect', 'NULL')
                category = item.get('Category', raw_aspect)
                aspect = raw_aspect if raw_aspect != "NULL" else category

                try:
                    v, a = map(float, item['VA'].split('#'))
                except:
                    continue

                enc = tokenizer(
                    text,
                    aspect,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )

                self.encodings["input_ids"].append(enc["input_ids"])
                self.encodings["attention_mask"].append(enc["attention_mask"])
                self.labels.append([v, a])

        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": self.labels[idx]
        }

    def to_pandas(self, batch_size=32):
        rows = []
        loader = DataLoader(self, batch_size=batch_size, shuffle=False)
        for batch in loader:
            input_ids_list = batch["input_ids"].numpy().tolist()
            attention_mask_list = batch["attention_mask"].numpy().tolist()
            labels_list = batch["labels"].numpy().tolist()

            for i in range(len(input_ids_list)):
                rows.append({
                    "input_ids": input_ids_list[i],
                    "attention_mask": attention_mask_list[i],
                    "labels": labels_list[i]
                })

        return pd.DataFrame(rows)


class DebertaRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.deberta = DebertaV2Model.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(self.deberta.config.hidden_size)
        self.regression_head = nn.Linear(self.deberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        normed_output = self.layer_norm(pooled_output)
        logits = self.regression_head(self.dropout(normed_output))
        return torch.sigmoid(logits) * 8.0 + 1.0
