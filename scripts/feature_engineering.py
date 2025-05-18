import pandas as pd
import numpy as np
import pandas_ta as ta
import os

def ensure_data_dir():
    path = os.path.abspath(os.path.join('airflow_projects', 'stock-etl-pipeline', 'data'))
    os.makedirs(path, exist_ok = True)
    return path
    
def generate_technical_indicators(table_name):
    """
    Enhances a DataFrame of financial time series data by adding a wide range of technical indicators.

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].

    Returns:
    pd.DataFrame: DataFrame with added technical indicators.
    """

    path = ensure_data_dir()

    df = pd.read_csv(f'{path}/processed/{table_name}.csv')

    # Utility indicators
    df['hl2'] = (df['high'] + df['low']) / 2
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    price_sources = ['close', 'hl2', 'hlc3', 'ohlc4']
    rsi_lengths = [6, 14, 21, 30, 50]
    ema_sma_lengths = [10, 20, 50, 100, 200]
    momentum_periods = [1, 2, 3, 5, 7, 9, 14, 21, 30, 50, 100]

    # Overlap indicators
    for col in price_sources:
        for length in ema_sma_lengths:
            df[f'ema_{length}_{col}'] = ta.ema(df[col], length)
            df[f'sma_{length}_{col}'] = ta.sma(df[col], length)

    bb = ta.bbands(df['close'], length=20)
    df = pd.concat([df, bb], axis=1)

    # Statistics Indicator
    df['corr_close_volume'] = df['close'].rolling(20).corr(df['volume'])

    # Volatility Indicator
    df['bbp'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])

    # Momentum indicators
    for col in price_sources:
        for length in rsi_lengths:
            df[f'rsi_{length}_{col}'] = ta.rsi(df[col], length)

    df = pd.concat([df, ta.macd(df['close'])], axis=1)
    df = pd.concat([df, ta.stochrsi(df['close'])], axis=1)
    df['willr'] = ta.willr(df['high'], df['low'], df['close'])

    stoch = ta.stoch(df['high'], df['low'], df['close'])
    df['K'] = stoch['STOCHk_14_3_3']
    df['D'] = stoch['STOCHd_14_3_3']
    df['J'] = 3 * df['K'] - 2 * df['D']

    df['bop'] = ta.bop(df['open'], df['high'], df['low'], df['close'])

    for p in momentum_periods:
        df[f'close_return_{p}'] = df['close'].pct_change(p)
        df[f'volume_change_{p}'] = df['volume'].pct_change(p)
        df[f'momentum_{p}'] = df['close'] - df['close'].shift(p)
        df[f'rolling_mean_{p}'] = df['close'].rolling(p).mean()
        if p != 1:
            df[f'rolling_std_{p}'] = df['close'].rolling(p).std()
            df[f'zscore_{p}'] = (df['close'] - df[f'rolling_mean_{p}']) / df[f'rolling_std_{p}']

    # Volume Indicators
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['pvr'] = df['close_return_1'] * df['volume_change_1']
    df['aobv'] = df['obv'].rolling(5).mean()

    # Trend Indicator
    df['ttm_trend'] = np.where(df['ema_20_close'] > df['ema_50_close'], 1, 0)

    df = pd.concat([
        df,
        ta.adx(df['high'], df['low'], df['close']),
        ta.cci(df['high'], df['low'], df['close']),
        ta.cmo(df['close']),
        ta.mfi(df['high'], df['low'], df['close'], df['volume']),
        ta.roc(df['close']),
        ta.trix(df['close']),
        ta.uo(df['high'], df['low'], df['close']),
        ta.wma(df['close'])
    ], axis=1)

    # Candle indicators
    df['inc'] = np.where(df['close'] > df['open'], 1, 0)
    df['dec'] = np.where(df['close'] < df['open'], 1, 0)
    df['cdl_doji'] = ta.cdl_doji(df['open'], df['high'], df['low'], df['close'])
    df['price_range'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['range_ratio'] = df['body'] / (df['price_range'] + 1e-9)

    # Class assignment
    df['class'] = np.where(df['open'].diff() > 0, 1, -1)
    df.dropna(inplace=True)

    df.to_csv(f'{path}/processed/{table_name}_indicators.csv', index = False)

    return df
