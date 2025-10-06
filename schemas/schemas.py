from pydantic import BaseModel
from typing import List, Optional

class InputData(BaseModel):
    hpo_ids: List[str]
    category: Optional[str] = None  # Only used in PUT


class PredictionResponse(BaseModel):
    prediction: List[str]
    cached: bool = False
    model_version: Optional[str] = None # Only used when inferenced from model


class StatusResponse(BaseModel):
    status: str

