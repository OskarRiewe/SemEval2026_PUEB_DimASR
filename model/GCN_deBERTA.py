import torch.nn as nn
from torch.utils.data import Dataset
from transformers import DebertaV2Model
import spacy
import torch
import numpy as np
import os


def get_spacy_nlp(lang_code):
    lang_map = {
        'eng': 'en_core_web_sm',
        'ukr': 'uk_core_news_sm',
        'tat': 'xx_ent_wiki_sm',
        'jpn': 'ja_core_news_sm',
        'zho': 'zh_core_web_sm',
        'rus': 'ru_core_news_sm'
    }
    
    model_name = lang_map.get(lang_code, 'en_core_web_sm')
    
    try:
        return spacy.load(model_name)
    except OSError:
        os.system(f"python -m spacy download {model_name}")
        return spacy.load(model_name)

class DimABSA_GCN_Dataset(Dataset):
    def __init__(self, data_list, tokenizer, lang, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_list = []
        nlp = get_spacy_nlp(lang)

        for data in data_list:
            text = data.get('Text', '')
            items = data.get('Quadruplet') or data.get('Aspect_VA')
            if not items: continue

            doc = nlp(text)

            for item in items:
                raw_aspect  = item.get('Aspect', 'NULL')
                category    = item.get('Category', raw_aspect)
                aspect      = raw_aspect if raw_aspect != "NULL" else category
                try:
                    v, a = map(float, item['VA'].split('#'))
                except: continue

                enc = tokenizer(
                    text, aspect,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_offsets_mapping=True
                )

                adj = self._create_adj_matrix(enc['offset_mapping'], doc)

                self.data_list.append({
                    "input_ids":        torch.tensor(enc["input_ids"]),
                    "attention_mask":   torch.tensor(enc["attention_mask"]),
                    "adj_matrix":       torch.tensor(adj, dtype=torch.float),
                    "labels":           torch.tensor([v, a], dtype=torch.float)
                })

    def _create_adj_matrix(self, offset_mapping, doc):
        adj = np.eye(self.max_length) # Self-loops

        token_to_word = []
        active_tokens = []
        for i, (start, end) in enumerate(offset_mapping):
            if start == end == 0 and i > 0: # Padding tokens
                token_to_word.append(-1)
                continue
            
            active_tokens.append(i)
            if start == end == 0: # Special tokens (like CLS)
                token_to_word.append(-1)
            else:
                found = -1
                for j, word in enumerate(doc):
                    if word.idx <= start < (word.idx + len(word.text)):
                        found = j
                        break
                token_to_word.append(found)

        # Global Connectivity for CLS (Index 0)
        # Connect CLS to ALL active tokens in the sentence
        for t_idx in active_tokens:
            if t_idx != 0:
                adj[0, t_idx] = 1
                adj[t_idx, 0] = 1

        for i, word in enumerate(doc):
            head_idx = word.head.i
            subwords_i = [idx for idx, w_idx in enumerate(token_to_word) if w_idx == i]
            subwords_head = [idx for idx, w_idx in enumerate(token_to_word) if w_idx == head_idx]

            for s_i in subwords_i:
                for s_h in subwords_head:
                    if s_i < self.max_length and s_h < self.max_length:
                        adj[s_i, s_h] = 1
                        adj[s_h, s_i] = 1
        return adj

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Use bias=True for better stability
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, adj):
        residual = x
        x = x.to(self.linear.weight.dtype)

        degree = torch.sum(adj, dim=-1)

        d_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)

        norm_adj = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        support = self.linear(x)
        output = torch.matmul(norm_adj, support)
        
        return torch.nn.functional.relu(output + residual)

class DebertaGCNRegressor(nn.Module):
    def __init__(self, model_name, gcn_layers=2):
        super().__init__()
        self.deberta = DebertaV2Model.from_pretrained(model_name)
        hidden_size = self.deberta.config.hidden_size

        self.gcns = nn.ModuleList([
            GCNLayer(hidden_size, hidden_size) for _ in range(gcn_layers)
        ])

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.regression_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, adj_matrix):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state

        for gcn in self.gcns:
            x = gcn(x, adj_matrix)

        pooled_output = x[:, 0, :]
        normed_output = self.layer_norm(pooled_output)
        logits = self.regression_head(self.dropout(normed_output))
        return torch.sigmoid(logits) * 8.0 + 1.0
