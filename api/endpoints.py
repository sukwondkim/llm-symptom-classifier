from fastapi import APIRouter, Depends, HTTPException

from api.inference_instance import inference_model
from schemas.schemas import PredictionResponse, StatusResponse
from utils.preprocessing import preprocess_input, preprocess_input_put

router = APIRouter()

@router.post("/post", response_model=PredictionResponse)
def predict(data: dict = Depends(preprocess_input)):
    cached: str = inference_model.query_cache(data["hpo_ids"])
    if cached:
        return PredictionResponse(prediction=[cached], cached=True)
    
    try:
        predicted: str = inference_model.predict(data["hpo_ids"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal model error: {e}")
    
    return PredictionResponse(prediction=predicted, model_version=inference_model.model_version)


@router.get("/get", response_model=PredictionResponse)
def get_data(data: dict = Depends(preprocess_input)):
    cached: str = inference_model.query_cache(data["hpo_ids"])
    if not cached:
        raise HTTPException(status_code=404, detail="Record not found")
    
    return PredictionResponse(prediction=cached, cached=True)


@router.put("/put", response_model=StatusResponse)
def set_data(data: dict = Depends(preprocess_input_put)):
    try:
        status = inference_model.set_cache(data["hpo_ids"], data["category"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    return StatusResponse(status=status)


@router.delete("/delete", response_model=StatusResponse)
def delete_data(data: dict = Depends(preprocess_input)):
    try:
        status = inference_model.delete_cache(data["hpo_ids"])
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    
    return StatusResponse(status=status)