from fastapi import APIRouter, Depends, HTTPException

from main import inference_model
from schemas import InputData, PredictionResponse, StatusResponse
from utils.preprocessing import preprocess_input

router = APIRouter()

@router.post("/post", response_model=PredictionResponse)
async def predict(data: dict = Depends(preprocess_input)):
    cached: str = await inference_model.query_cache(data["hpo_ids"])
    if cached:
        return PredictionResponse(prediction=[cached], cached=True)
    
    try:
        predicted: list = inference_model.predict(data["hpo_ids"])
    except Exception:
        raise HTTPException(status_code=500, detail="Internal model error")
    
    return PredictionResponse(prediction=predicted, model_version=inference_model.model_version)


@router.get("/get", response_model=PredictionResponse)
async def get_data(data: dict = Depends(preprocess_input)):
    cached: str = await inference_model.query_cache(data["hpo_ids"])
    if not cached:
        raise HTTPException(status_code=404, detail="Record not found")
    
    return PredictionResponse(prediction=[cached], cached=True)


@router.put("/put", response_model=StatusResponse)
async def set_data(data: dict = Depends(preprocess_input)):
    try:
        status = await inference_model.set_cache(data["hpo_ids"], data["category"])
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    return StatusResponse(status=status)


@router.delete("/delete", response_model=StatusResponse)
async def delete_data(data: dict = Depends(preprocess_input)):
    try:
        status = await inference_model.delete_cache(data["hpo_ids"])
    except ValueError as ve:
        raise HTTPException(status_code=404, detail="Record not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    
    return StatusResponse(status=status)