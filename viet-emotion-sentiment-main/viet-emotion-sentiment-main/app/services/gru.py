import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional, Tuple
from app.utils.text import normalize_text

from app.core.config import settings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GRUModel:
    def __init__(self):
        self.model_dir = "models/gru"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.config = AutoConfig.from_pretrained(str(self.model_dir), trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, config=self.config, trust_remote_code=True, local_files_only=True)
        self.use_half = bool(settings.use_half) and DEVICE == "cuda"
        self.model.to(DEVICE).eval()
        self.labels = self.model.config.id2label

    def _enc(self, text: str) -> dict:
        return self.tokenizer(text, truncation=True, padding="max_length", max_length=self.config.max_position_embeddings, return_tensors="pt")

    def predict(
        self, texts: List[str], return_probs: bool = True
    ) -> Tuple[List[str], List[str], Optional[List[Dict[str, float]]]]:
        if not texts:
            return [], [], [] if return_probs else ([], [], None)

        tokens = [normalize_text(t) for t in texts]
        exps = [normalize_text(t, use_vitok=False, join_negation_flag=False) for t in texts]
        batch = self._enc(tokens)
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

        with torch.no_grad():
            if self.use_half:
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                    logits = self.model(**batch).logits
            else:
                logits = self.model(**batch).logits

            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            labels = [self.labels[i] for i in pred_ids]

            if not return_probs:
                return exps, labels, None

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        probs_list: List[Dict[str, float]] = []
        for row in probs:
            probs_list.append({
                self.labels[i]: float(round(float(row[i]), 4))
                for i in range(len(self.labels))
            })
            
        return exps, labels, probs_list