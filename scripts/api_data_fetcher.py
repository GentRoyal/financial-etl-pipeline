import pandas as pd
import os
import requests
from io import StringIO
# from config import settings
from scripts.config import settings

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
        
        absolute_path = os.path.abspath(__file__)
        directory_name = os.path.dirname(absolute_path)
        parent_name = os.path.dirname(directory_name)
        self.path = os.path.join(parent_name, 'data', 'raw')
    
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def get_stock(self, symbol, size = "compact"):
        """
        Fetches daily stock data for a given symbol and saves it as a CSV file.

        Parameters:
        symbol (str): The stock ticker symbol.
        size (str): The amount of data to fetch ('compact' or 'full').

        Returns:
        pd.DataFrame: The stock data as a DataFrame or an empty DataFrame if an error occurs.
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": size,
            "apikey": self.api_key,
            "datatype": "csv"
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df.set_index('timestamp', inplace=True)
            df.sort_index(ascending=False)

            df.to_csv(f'{self.path}/{symbol}.csv')

            return df

        except requests.exceptions.RequestException as req_err:
            print(f"Request error while fetching stock data: {req_err}")
        except pd.errors.ParserError as parse_err:
            print(f"Parsing error while reading stock data: {parse_err}")
        except Exception as e:
            print(f"Unexpected error fetching stock data: {e}")

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