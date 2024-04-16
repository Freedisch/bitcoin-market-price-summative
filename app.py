from fastapi import FastAPI
from pydantic import BaseModel
from src.model import predict_pipeline
from src.model import __version__ as model_version


app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    prices: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    prices = predict_pipeline(payload.text)
    return {"prices": prices}