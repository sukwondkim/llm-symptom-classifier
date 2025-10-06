from typing import List, Optional

import torch
from redis import Redis
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.preprocessing import make_query, categories

class InferenceModel:
    def __init__(self):
        self.redis_client: Optional[Redis] = None
        self.tokenizer = None
        self.base_model = "microsoft/biogpt"
        self.model = None
        self.model_version = "v1.0"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.id2label = {i: label for i, label in enumerate(categories)}
        self.label2id = {label: i for i, label in enumerate(categories)}
    
    def setup_redis(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis_client = Redis(host=host, port=port, db=db)

    def close_redis(self):
        self.redis_client.bgsave()
        self.redis_client.close()

    def load_model(self, finetuned_model_path: str = "models/best_model.pth"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels = len(categories),
            torch_dtype=torch.float32
        )

        self.model = PeftModel.from_pretrained(base_model, finetuned_model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model successfully loaded from {finetuned_model_path}")
    
    def _make_cache_key(self, hpo_ids: List[str]):
        return "|".join(hpo_ids)

    def query_cache(self, hpo_ids: List[str]):
        key = self._make_cache_key(hpo_ids)
        return self.redis_client.get(key)
    
    def set_cache(self, hpo_ids: List[str], category: str):
        key = self._make_cache_key(hpo_ids)
        exists = self.redis_client.exists(key)
        self.redis_client.set(key, category)
        return "Updated" if exists else "Created"
    
    def delete_cache(self, hpo_ids: List[str]):
        key = self._make_cache_key(hpo_ids)
        deleted = self.redis_client.delete(key)
        if not deleted:
            raise ValueError("Record not found")
        
        return "Deleted"
    
    def predict(self, hpo_ids: List[str]):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        system_query = make_query(hpo_ids)
        inputs = self.tokenizer(system_query, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        predicted_class_id = logits.argmax(dim=-1).item()
        predicted_label = self.id2label[predicted_class_id]

        return predicted_label
