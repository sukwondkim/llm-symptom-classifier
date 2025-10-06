from pydantic import BaseModel
from typing import List, Optional

class InputDataBase(BaseModel):
    hpo_ids: List[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "hpo_ids": ["HP:0000641", "HP:0001250"],
            }
        }
    }
    
class InputDataPut(BaseModel):
    hpo_ids: List[str]
    category: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "hpo_ids": ["HP:0000641", "HP:0001250"],
                "category": "Neurodevelopmental disorder",
            }
        }
    }
    


class PredictionResponse(BaseModel):
    prediction: str
    cached: bool = False
    model_version: Optional[str] = None # Only used when inferenced from model
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": "Cardiovascular disorder",
                "cached": True,
                "model_version": "v1.0"
            }
        }
    }


class StatusResponse(BaseModel):
    status: str

    model_config = {
        "json_schema_extra": {
            "example": {"status": "Created"}
        }
    }
