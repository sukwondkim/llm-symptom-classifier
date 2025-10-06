from typing import List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils.preprocessing import make_query

class HPODataset(Dataset):
    def __init__(
        self,
        hpo_ids: List[List[str]],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_token_len: int,
    ):
        if len(hpo_ids) != len(labels):
            raise ValueError(f"Length of hpo_ids {len(hpo_ids)} and labels {len(labels)} must match.")
        
        self.hpo_ids = hpo_ids
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        query = make_query(self.hpo_ids[idx])
        label = self.labels[idx]

        inputs = self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=self.max_token_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }
