import pandas as pd
import os
import requests
from io import StringIO
import logging

from scripts.config import settings

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MyAPI:
    """
    A class for fetching stock and cryptocurrency data from the Alpha Vantage API.
    """

    def __init__(self):
        """
        Initializes the API with a base URL and API key.
        Creates the data directory if it doesn't exist.
        """
        self.api_key = settings.api_key
        self.base_url = "https://www.alphavantage.co/query"
        
        self.absolute_path = os.path.abspath(__file__)
        self.directory_name = os.path.dirname(self.absolute_path)
        self.parent_name = os.path.dirname(self.directory_name)
        self.base_path = None

    def get_stock(self, symbol, size = "compact"):
        """
        Fetches daily stock data for a given symbol and saves it as a CSV file.
        Parameters:
        symbol (str): The stock ticker symbol.
        size (str): The amount of data to fetch ('compact' or 'full').
        Returns:
        pd.DataFrame: The stock data as a DataFrame or an empty DataFrame if an error occurs.
        """
        env = os.getenv("ENVIRONMENT", "local")
        logger.info(f'Working Environment: {env}')

        if env == "docker":
            self.base_path = "/app/app"
            
        else:
            self.base_path = self.parent_name
        
        self.path = os.path.join(self.base_path, 'data', 'raw')
    
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": size,
            "apikey": self.api_key,
            "datatype": "csv"
        }
        try:
            logger.info(f"Fetching stock data for symbol: {symbol}")
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df.set_index('timestamp', inplace = True)
            df.sort_index(ascending = False)

            # Ensure directory exists
            if not os.path.exists(self.path):
                os.makedirs(self.path, exist_ok = True)

            csv_file_path = os.path.join(self.path, f'{symbol}.csv')

            df.to_csv(csv_file_path)
            logger.info(f"Successfully fetched and saved stock data for {symbol} at {csv_file_path}")

            
            return df
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error while fetching stock data for {symbol}: {req_err}")
            
        except pd.errors.ParserError as parse_err:
            logger.error(f"Parsing error while reading stock data for {symbol}: {parse_err}")
            
        except Exception as e:
            logger.error(f"Unexpected error fetching stock data for {symbol}: {e}")
            
        return pd.DataFrame()

    def get_crypto(self, symbol, market, size="compact"):
        """
        Fetches daily cryptocurrency data for a given symbol and market, and saves it as a CSV file.

        Parameters:
        symbol (str): The cryptocurrency symbol (e.g., 'BTC').
        market (str): The market currency (e.g., 'USD').
        size (str): The amount of data to fetch ('compact' or 'full').

        Returns:
        pd.DataFrame: The crypto data as a DataFrame or an empty DataFrame if an error occurs.
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "market": market,
            "outputsize": size,
            "apikey": self.api_key,
            "datatype": "csv"
        }

        env = os.getenv("ENVIRONMENT", "local")
        logger.info(f'Working Environment: {env}')
    
        if env == "docker":
            self.base_path = "/app/app"
                
        else:
            self.base_path = self.parent_name
            
        self.path = os.path.join(self.base_path, 'data', 'raw')
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)


        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df.set_index('timestamp', inplace=True)
            df.sort_index(ascending=False)

            df.to_csv(f'{self.path}/{symbol}_{market}.csv')

            return df

        except requests.exceptions.RequestException as req_err:
            print(f"Request error while fetching crypto data: {req_err}")
        except pd.errors.ParserError as parse_err:
            print(f"Parsing error while reading crypto data: {parse_err}")
        except Exception as e:
            print(f"Unexpected error fetching crypto data: {e}")

        return pd.DataFrame()