from contextlib import asynccontextmanager
from typing import Annotated, List

from fastapi import FastAPI, Query
from pydantic import BaseModel

from model import *
from logger_common import *

class PredictionInput(BaseModel):
    vtype: str
    x: list[float] | list[list[float]] | None
    steps: int = 1
    default_zero: bool = True


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
    if (data.x is None or len(data.x) == 0 or len(data.x[0]) == 0) and data.default_zero:
        logger.info('Default to zero-tensor inputs as inputs are empty and default_zero is enabled')
        logger.info(f'Input shape before using default value: {np.array(data.x).shape}')
        data.x = np.zeros((len(data.x), sizes[0]))
        logger.info(f'Input shape after using default value: {data.x.shape}')
    result = predict(models[data.vtype], Scaler(), data.x, data.steps)
    return {"result": result.tolist(), "status": "successful", "message":""}

@app.get("/health")
async def health():
    return {"status": "healthy"}

