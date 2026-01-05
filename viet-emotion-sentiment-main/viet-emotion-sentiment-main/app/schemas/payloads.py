# app/schemas/payloads.py
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class PredictIn(BaseModel):
    text: str = Field(..., description="Câu cần phân tích cảm xúc")
    return_probs: bool = True

class PredictOut(BaseModel):
    expanded: str
    label: str
    probs: Optional[Dict[str, float]] = None

class PredictBatchIn(BaseModel):
    texts: List[str]
    batch_size: int = 64
    return_probs: bool = False

class PredictBatchOut(BaseModel):
    expanded: List[str]
    labels: List[str]
    probs: Optional[List[Dict[str, float]]] = None
