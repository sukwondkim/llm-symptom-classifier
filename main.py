from contextlib import asynccontextmanager
from fastapi import FastAPI

from models.inference import InferenceModel
from api.endpoints import router as api_router

inference_model = InferenceModel()

@asynccontextmanager
def lifespan(app: FastAPI):
    print("[Startup] Connecting to Redis...")
    inference_model.setup_redis()

    print("[Startup] Loading a model...")
    inference_model.load_model()

    yield

    print("[Shutdown] Disconnecting to Redis...")
    inference_model.close_redis()

app = FastAPI(title="LLM Symptom Classifier API", version="1.0", lifespan=lifespan)
app.include_router(api_router, prefix="/v1")
