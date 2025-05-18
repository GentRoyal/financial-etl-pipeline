import pandas as pd
import os
import requests
from io import StringIO

def ensure_data_dir():
    #path = os.path.abspath(os.path.join('..', 'data', 'raw'))
    path = os.path.abspath(os.path.join('airflow_projects', 'stock-etl-pipeline', 'data', 'raw'))
    os.makedirs(path, exist_ok = True)
    return path

def get_stock(symbol, size = "compact"):
    api_key = "d67167cc02e0a4127df1081013f93a8ff17f7b38f592d47e5b9d8d9049952bc7aff529412868a67827302fdd3756570413848c524d33a8ce2101e9f80e21550c65c8a4e50e628f07f6f4981f38d0d6f9ee6306ac47969cbf0ffa8f5fbe364e579c9a2f36a78fb7d9ed4e15f19bda0e61235f1408f7c0dfd5ca58096f6ac82bfd"
    #"8b6799158085376518839527a7d3695375bf5ed0dbd7827f07432d1727b80d37cabd96fb6d26794d2de7e5f057c54ab86b7d139761cb07bb164c8d37e68ca53fb15fc5bd527b68716bb455b62d092fa54675fe6622fcd06808778b9cea35afba7ebb0c9aa40fe6ae8303785b5e0b89920659d9f099bdc57af1413722525aea78"
    base_url = "https://www.alphavantage.co/query"
    path = ensure_data_dir()

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": size,
        "apikey": api_key,
        "datatype": "csv"
    }

    try:
        response = requests.get(base_url, params = params, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df.set_index('timestamp', inplace=True)
        df.sort_index(ascending=False, inplace=True)

        df.to_csv(f'{path}/{symbol}.csv')

        return df

    except requests.exceptions.RequestException as req_err:
        print(f"Request error while fetching stock data: {req_err}")
        
    except pd.errors.ParserError as parse_err:
        print(f"Parsing error while reading stock data: {parse_err}")
        
    except Exception as e:
        print(f"Unexpected error fetching stock data: {e}")

    return pd.DataFrame()

def get_crypto(symbol, market, size = "compact"):
    api_key = "d67167cc02e0a4127df1081013f93a8ff17f7b38f592d47e5b9d8d9049952bc7aff529412868a67827302fdd3756570413848c524d33a8ce2101e9f80e21550c65c8a4e50e628f07f6f4981f38d0d6f9ee6306ac47969cbf0ffa8f5fbe364e579c9a2f36a78fb7d9ed4e15f19bda0e61235f1408f7c0dfd5ca58096f6ac82bfd"
    base_url = "https://www.alphavantage.co/query"
    path = ensure_data_dir()

    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": symbol,
        "market": market,
        "apikey": api_key,
        "datatype": "csv"
    }

    try:
        response = requests.get(base_url, params = params, timeout = 10)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df.set_index('timestamp', inplace=True)
        df.sort_index(ascending=False, inplace=True)

        df.to_csv(f'{path}/{symbol}_{market}.csv')

        return df

    except requests.exceptions.RequestException as req_err:
        print(f"Request error while fetching crypto data: {req_err}")
        
    except pd.errors.ParserError as parse_err:
        print(f"Parsing error while reading crypto data: {parse_err}")
        
    except Exception as e:
        print(f"Unexpected error fetching crypto data: {e}")

    return pd.DataFrame()