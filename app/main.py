from contextlib import asynccontextmanager
from typing import Annotated, List

from fastapi import FastAPI, Query
from pydantic import BaseModel

from model import *

class PredictionInput(BaseModel):
    vtype: str
    x: list[float] | list[list[float]]
    steps: int = 1


models = {
    "bus": "bus.pt",
    "car": "car.pt",
    "motorbike": "motorbike.pt",
    "truck": "truck.pt",
    "container": "container_truck.pt",
    "pedestrian": "pedestrian.pt",
    "other_vehicle": "vehicle_others.pt",
}

sizes = [16, 64, 32, 1]
dropout_rate = 0.2

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    for key in models.keys():
        model = MLP(sizes, dropout_rate)
        load_model(model, models[key])
        models[key] = model
    yield
    # Clean up the ML models and release the resources
    models.clear()
    
app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def run_predict(data: PredictionInput):
    if data.vtype not in models:
        return {"result": -1, "status": "failed", "message": f"No model with type: {data.vtype}"}
    result = predict(models[data.vtype], Scaler(), data.x, data.steps)
    return {"result": result.tolist(), "status": "successful", "message":""}

@app.get("/health")
async def health():
    return {"status": "healthy"}
