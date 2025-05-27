from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import pickle as pkl
import os
import logging
import pandas as pd
import numpy as np
import sys

sys.path.append(os.path.join(os.environ.get('AIRFLOW_HOME'), 'stock-etl-pipeline'))

from scripts.api_data_fetcher import MyAPI
from scripts.database_handler import MyDBRepo

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title = "Trading Model Prediction API",
    description = "API for generating predictions using a pre-trained trading model",
    version = "1.0.0"
)

REQUIRED_FEATURES = ['rsi_6_hl2', 'momentum_3', 'close_return_2', 'rsi_6_close', 'rsi_21_hl2', 'rsi_21_hlc3', 'rsi_14_hlc3', 'CCI_14_0.015', 'zscore_7', 'momentum_1', 'zscore_3', 'rsi_21_ohlc4', 'bbp', 'zscore_30', 'zscore_5', 'rsi_14_close', 'pvr', 'ADX_14', 'J', 'dec', 'rsi_14_hl2', 'inc', 'rsi_6_ohlc4', 'rolling_std_30', 'CMO_14', 'momentum_7', 'rsi_6_hlc3', 'zscore_2', 'zscore_9', 'zscore_14', 'rolling_std_9', 'rsi_14_ohlc4', 'DMN_14', 'BBP_20_2.0', 'zscore_21', 'willr', 'rsi_30_hl2', 'bop', 'close_return_1', 'close_return_3', 'DMP_14', 'rsi_30_ohlc4', 'momentum_2', 'cdl_doji']

model_accuracy, model = None, None
file = os.path.abspath(__file__)
directory = os.path.dirname(file)
parent = os.path.dirname(directory)

class DataPoint(BaseModel):
    features : dict[str, float]

    @validator('features')
    def validate_features(cls, features):
        missing_feature = [feat for feat in REQUIRED_FEATURES if feat not in features]

        if missing_feature:
            raise ValueError(f'Missing Feature: {missing_feature}')
            logger.error(f'Missing Feature: {missing_feature}')

        return features

class PredictionResponse(BaseModel):
    predictions : int
    probabilities : dict[str, float] | None

@app.get('/')
async def root():
    return {'message' : 'Application Running'}


@app.on_event('startup')
async def load_model():
    global model_accuracy, model

    try:
        model_path = os.path.join(parent, 'models', 'model_20250521_074410.pkl')
        result_path = os.path.join(parent, 'data', 'results', 'results_20250521_074410.csv')

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pkl.load(f)
            logger.info(f'{model_path} loaded successfully')

        if os.path.exists(result_path):
            df = pd.read_csv(result_path)
            model_accuracy = np.max(df['accuracy']) * 100
            
            logger.info(f'Model Accuracy: {model_accuracy:.2f}')

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

@app.post('/predict', response_model = PredictionResponse)
async def make_prediction(data : DataPoint):
    if model is None:
        logger.info('Model has not been loaded!')
        return JSONResponse(status_code = 503, content = {"message": "Model not available yet."})

    input_data = pd.DataFrame([data.features])
    
    input_data = input_data[REQUIRED_FEATURES]

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    probabilities = dict(zip(map(str, model.classes_), probabilities))

    logger.info(f"Prediction: {prediction}")
    logger.info(f"Probability: {probabilities}")

    return {
        'predictions' : prediction,
        'probabilities' : probabilities

    }

@app.get('/model_accuracy')
async def get_accuracy():
    if model_accuracy is None:
        logger.info('Model Accuracy has not been calculated')
        return JSONResponse(status_code = 503, content = {"message": "Model Accuracy not available yet."})

    return np.round(model_accuracy, 2)


@app.get('/load-data', response_model=list)
async def load_data(symbol : str = Query(..., description = "Stock symbol to load data for")):
    try:
        myapi = MyAPI()
        df = myapi.get_stock(symbol, size = "full")

        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return JSONResponse(
                status_code = 404,
                content={"message": f"No data found for symbol '{symbol}'"}
            )

        logger.info(f"Stock data loaded successfully for {symbol}")
        return JSONResponse(
            status_code = 200,
            content = df.to_dict(orient = "records")
        )

    except Exception as e:
        logger.error(f"Failed to load stock data for {symbol}: {str(e)}")
        return JSONResponse(
            status_code = 503,
            content = {"message": f"Service unavailable. Could not load data for '{symbol}'."}
        )