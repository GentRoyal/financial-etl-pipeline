from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from pydantic import BaseModel, validator

import os
import sys
import pickle as pkl
import pandas as pd
import numpy as np
import ast

import sys
sys.path.append('./app')

from scripts.api_data_fetcher import MyAPI
from scripts.feature_engineering import TechnicalIndicatorGenerator

import logging
logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title = 'Trading Model Prediction API',
    description = "API for generating predictions using a pre-trained trading model",
    version = "1.0.0"
)


@app.get('/')
async def root():
    return {"message" : "Application is Up and Running"}

print('Trading Model Prediction API')

class DataPoint(BaseModel):
    features : dict[str, float]

    @validator('features')
    def validate_features(features):
        missing_features = [feat for feat in REQUIRED_FEATURES if feat not in features]

        if missing_features:
            raise ValueError(f'Missing Features: {missing_features}')
            logger.error(f'Missing Features: {missing_features}')

        logger.info(f'Features Loaded Successfully')
        return features

class PredictionResponse(BaseModel):
    prediction : int
    probabilities : dict[str, float] | None

model, model_accuracy, REQUIRED_FEATURES = None, None, None
file = os.path.abspath(__file__)
parent = os.path.dirname(file)

@app.on_event('startup')
async def load_application_dependencies():
    global model, model_accuracy, REQUIRED_FEATURES

    try:
        model_path = os.path.join(parent, 'models', 'model_20250521_074410.pkl')
        result_path = os.path.join(parent, 'data', 'results', 'results_20250521_074410.csv')

        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pkl.load(f)
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.warning(f"Model path not found: {model_path}")

        if os.path.exists(result_path):
            df = pd.read_csv(result_path)
            model_accuracy = df['accuracy'].max()
            
            features_str = df.loc[df['accuracy'] == model_accuracy, 'features'].values[0]
            REQUIRED_FEATURES = ast.literal_eval(features_str)

            first_five = REQUIRED_FEATURES[:5]
            first_five[-1] = first_five[-1] + '...'
            
            model_accuracy *= 100

            logger.info(f"Model results loaded from {result_path}")
            logger.info(f"Best model accuracy: {model_accuracy:.2f}%")
            logger.info(f"Required features for best model: {first_five}")
        else:
            logger.warning(f"Result path not found: {result_path}")

    except Exception as e:
        logger.exception("An error occurred while loading the model or results")
        raise ValueError(f"Failed to load model or results: {str(e)}")

@app.get('/model_accuracy')
async def get_model_accuracy():
    if model_accuracy is None:
        logger.error("Model accuracy not available. Make sure the model is loaded.")
        return JSONResponse(
            status_code = 503,
            content = {"error": "Model accuracy not available. Try again later."}
        )
    
    return {"model_accuracy": f'{model_accuracy:.2f}%'}

@app.post('/predict_stock_direction', response_model = PredictionResponse)
async def make_predictions_from_input(data : DataPoint):
    if model is None:
        logger.error("Model not available. Make sure the model is loaded.")
        return JSONResponse(
            status_code = 503,
            content = {"error": "Model not available. Try again later."}
        )

    model_features = pd.DataFrame([data.features])
    model_features = model_features[REQUIRED_FEATURES]
    
    prediction = model.predict(model_features)[0]
    probabilities = model.predict_proba(model_features)[0]

    probabilities = dict(zip(map(str, model.classes_), probabilities))

    return {'prediction' : prediction, 'probabilities' : probabilities}

def fetch_stock_data(symbol: str, size: str):
    try:
        myapi = MyAPI()
        df = myapi.get_stock(symbol = symbol, size = size)

        if df.empty:
            return None, f"No data found for symbol '{symbol}'"
        
        return df, None
        
    except Exception as e:
        return None, str(e)

@app.post("/load_data_from_api")
def load_data_from_api(
    symbol: str = Query(..., description='Enter Stock Symbol (e.g. AAPL)'),
    size: str = Query(..., description='Enter Size (compact/full)')):

    try:
        df, error = fetch_stock_data(symbol, size)
        if error:
            logger.error(f"Failed to load stock data for {symbol}: {error}")
            return JSONResponse(status_code = 503, content = {"message": error})
    
        logger.info(f"Stock data loaded successfully for {symbol}")
        return JSONResponse(status_code = 200, content = df.to_dict(orient = "index"))
    
    except Exception as e:
        logger.error(f"Failed to load stock data for {symbol}: {str(e)}")
        return JSONResponse(
            status_code = 503,
            content = {"message": f"Service unavailable. Could not load data for '{symbol}'."}
        )
        
@app.post('/historical_prediction', response_model = PredictionResponse)
async def historical_prediction(symbol: str = Query(..., description = 'Enter Stock Symbol (e.g. AAPL)')):
    try:
        df, error = fetch_stock_data(symbol, size = "full")
        if error:
            logger.error(f"Failed to load stock data for {symbol}: {error}")
            return JSONResponse(status_code = 503, content = {"message": error})
    
        tech_indicator = TechnicalIndicatorGenerator()
        df = tech_indicator.generate_indicators(df, symbol)
        df = df[REQUIRED_FEATURES]

        indicators = df.columns[:5]
        
        logger.info(f"Technical Indicators Generated successfully: {indicators}")

        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
    
        probabilities = dict(zip(map(str, model.classes_), probabilities))
    
        return {'prediction' : prediction, 'probabilities' : probabilities}
        
    except Exception as e:
        logger.error(f"Failed to load stock data for {symbol}: {str(e)}")
        return JSONResponse(
            status_code = 503,
            content = {"message": f"Service unavailable. Could not load data for '{symbol}'."}
        )