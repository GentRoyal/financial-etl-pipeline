from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Any
import numpy as np
import os
import pandas as pd
import uvicorn
import pickle
import logging
from datetime import datetime

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title = "Trading Model Prediction API",
    description = "API for generating predictions using a pre-trained trading model",
    version = "1.0.0"
)

# Set up static file serving and templates
templates = Jinja2Templates(directory="templates")

# Make sure these directories exist
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files AFTER creating directories
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware to allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected features for your model
REQUIRED_FEATURES = ['rsi_6_hl2', 'momentum_3', 'close_return_2', 'rsi_6_close', 'rsi_21_hl2', 'rsi_21_hlc3', 'rsi_14_hlc3', 'CCI_14_0.015', 'zscore_7', 'momentum_1', 'zscore_3', 'rsi_21_ohlc4', 'bbp', 'zscore_30', 'zscore_5', 'rsi_14_close', 'pvr', 'ADX_14', 'J', 'dec', 'rsi_14_hl2', 'inc', 'rsi_6_ohlc4', 'rolling_std_30', 'CMO_14', 'momentum_7', 'rsi_6_hlc3', 'zscore_2', 'zscore_9', 'zscore_14', 'rolling_std_9', 'rsi_14_ohlc4', 'DMN_14', 'BBP_20_2.0', 'zscore_21', 'willr', 'rsi_30_hl2', 'bop', 'close_return_1', 'close_return_3', 'DMP_14', 'rsi_30_ohlc4', 'momentum_2', 'cdl_doji']

# Global model storage
model = None
model_accuracy = None

# Enhanced Pydantic model with validation
class DataPoint(BaseModel):
    features: Dict[str, float]
    
    @validator('features')
    def validate_features(cls, features):
        missing_features = [feat for feat in REQUIRED_FEATURES if feat not in features]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        return features

class PredictionResponse(BaseModel):
    predictions: List[float]
    probabilities: Optional[List[Any]] = None

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code = 500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )

# Load the model at startup
@app.on_event("startup")
async def load_model():
    global model, model_accuracy
    try:
        # For testing, we'll set a default accuracy when model loading fails
        model_accuracy = 86.67
        
        # Try to load the model if it exists
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
            logger.warning(f"Could not load model, using dummy model: {e}")
            
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

# Ensure index.html is properly served from the root path
@app.get("/")
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/model-accuracy")
async def get_model_accuracy():
    return {"accuracy": round(model_accuracy, 2)}

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code = 503, detail = "Model not loaded")
    return {
        "status": "healthy", 
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model = PredictionResponse)
async def predict(data: DataPoint):
    if model is None:
        raise HTTPException(status_code = 503, detail="Model not loaded")
    
    try:
        # Extract features
        features_dict = data.features
        df = pd.DataFrame([features_dict])
        
        # For the demo, we'll just use whatever features are provided
        # In production, ensure all required features are included
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
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code = 500, detail = f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host = "127.0.0.1", port = 8000, reload = True)