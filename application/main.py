import uvicorn
import pickle
import os
import numpy as np
import pandas as pd

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

import logging
logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title = "Trading Model Prediction API",
    description = "API for generating predictions using a pre-trained trading model",
    version = "1.0.0"
)

REQUIRED_FEATURES = ['rsi_6_hl2', 'momentum_3', 'close_return_2', 'rsi_6_close', 'rsi_21_hl2', 'rsi_21_hlc3', 'rsi_14_hlc3', 'CCI_14_0.015', 'zscore_7', 'momentum_1', 'zscore_3', 'rsi_21_ohlc4', 'bbp', 'zscore_30', 'zscore_5', 'rsi_14_close', 'pvr', 'ADX_14', 'J', 'dec', 'rsi_14_hl2', 'inc', 'rsi_6_ohlc4', 'rolling_std_30', 'CMO_14', 'momentum_7', 'rsi_6_hlc3', 'zscore_2', 'zscore_9', 'zscore_14', 'rolling_std_9', 'rsi_14_ohlc4', 'DMN_14', 'BBP_20_2.0', 'zscore_21', 'willr', 'rsi_30_hl2', 'bop', 'close_return_1', 'close_return_3', 'DMP_14', 'rsi_30_ohlc4', 'momentum_2', 'cdl_doji']

model = None
model_accuracy = None

class PredictionResponse(BaseModel):
    predictions : list[float]
    probabilities: list[Any] | None

class DataPoint(BaseModel):
    features : dict[str, float]

    @validator('features')
    def validate_features(cls, features):
        missing_features = [feature for feature in REQUIRED_FEATURES if feature not in features]

        if missing_features:
            logger.info(f'Missing Features {missing_features}')
            raise ValueError(f'Missing Features {missing_features}')

        return features

@app.get("/")
def root():
    return {"message": "Welcome to the Trading Model API"}

@app.on_event("startup")
async def load_model():
    global model, model_accuracy
    
    try:
        file = os.path.abspath(__file__)
        directory = os.path.dirname(file)
        parent = os.path.dirname(directory)
        
        result_path = os.path.join(parent, 'data', 'results', 'results_20250521_074410.csv')
        if os.path.exists(result_path):
            result_df = pd.read_csv(result_path)
            model_accuracy = np.max(result_df['accuracy']) * 100
    
        latest_model_path = os.path.join(parent, 'models', "model_20250521_074410.pkl")
        
        if os.path.exists(latest_model_path):
            with open(latest_model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded successfully from {latest_model_path}")
                
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {str(e)}")


@app.post("/predict", response_model = PredictionResponse)
async def get_predictions(data : DataPoint):
    if model is None:
        return JSONResponse(status_code = 503, content = {"message": "Model not available yet."})
    features_dict = data.features
    df = pd.DataFrame([features_dict])

    available_features = [f for f in REQUIRED_FEATURES if f in df.columns]
    input_data = df[available_features]
    
    # Fill missing values with zeros for demonstration
    for feature in REQUIRED_FEATURES:
        if feature not in input_data.columns:
            input_data[feature] = 0
        
    # Ensure all features are in the correct order
    input_data = input_data[REQUIRED_FEATURES]
        
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    class_labels = model.classes_.tolist()
    probabilities = [{str(label): float(p)} for label, p in zip(class_labels, proba)]

    return {
            "predictions": [float(prediction)], 
            "probabilities": probabilities
        }

@app.get("/accuracy")
async def get_accuracy():
    if model_accuracy is None:
        return JSONResponse(status_code = 503, content = {"message": "Model accuracy not available yet."})

    return {"model_accuracy": round(model_accuracy, 2)}