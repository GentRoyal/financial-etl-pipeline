import pandas as pd
import numpy as np
import pandas_ta as ta
import os

class TechnicalIndicatorGenerator:
    """
    Generates a wide range of technical indicators for financial time series data.
    """
    
    def __init__(self):
        absolute_path = os.path.abspath(__file__)
        directory_name = os.path.dirname(absolute_path)
        parent_name = os.path.dirname(directory_name)
        self.path = os.path.join(parent_name, 'data')        

    def generate_indicators(self, symbol):
        """
        Adds a variety of technical indicators to a DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].

        Returns:
        pd.DataFrame: Enhanced DataFrame with technical indicators.
        """

        df = pd.read_csv(f'{self.path}/processed/{symbol}.csv') 
        
        # Utility Indicators
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
        price_sources = ['close', 'hl2', 'hlc3', 'ohlc4']
        rsi_lengths = [6, 14, 21, 30, 50]
        ema_sma_lengths = [10, 20, 50, 100, 200]
        momentum_periods = [1, 2, 3, 5, 7, 9, 14, 21, 30, 50, 100]
    
    
        # Overlap
        for col in price_sources:
            for length in ema_sma_lengths:
                df[f'ema_{length}_{col}'] = ta.ema(df[col], length) # EMA
                df[f'sma_{length}_{col}'] = ta.sma(df[col], length)  # SMA
        
        bb = ta.bbands(df['close'], length=20) # Bollinger Bands and BBP
        df = pd.concat([df, bb], axis=1)
    
        # Statistics Indicator
        df['corr_close_volume'] = df['close'].rolling(20).corr(df['volume']) # Correlation trend
    
        # Volatility Indicator
        df['bbp'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        
        # Momentum Indicators
        for col in price_sources: # RSI
            for length in rsi_lengths:
                df[f'rsi_{length}_{col}'] = ta.rsi(df[col], length)
    
        macd = ta.macd(df['close']) # MACD
        df = pd.concat([df, macd], axis=1)
    
        stochrsi = ta.stochrsi(df['close']) # STOCH RSI
        df = pd.concat([df, stochrsi], axis=1)
    
        df['willr'] = ta.willr(df['high'], df['low'], df['close']) # Williams %R
    
        stoch = ta.stoch(df['high'], df['low'], df['close']) # KDJ
        df['K'], df['D'] = stoch['STOCHk_14_3_3'], stoch['STOCHd_14_3_3']
        df['J'] = 3 * df['K'] - 2 * df['D']
    
        df['bop'] = ta.bop(df['open'], df['high'], df['low'], df['close']) # Balance of Power
        
        for p in momentum_periods:
            df[f'close_return_{p}'] = df['close'].pct_change(p) # Statistics Indicator: Close Return
            df[f'volume_change_{p}'] = df['volume'].pct_change(p) # Volume Indicator: Volume Change
            df[f'momentum_{p}'] = df['close'] - df['close'].shift(p)
            df[f'rolling_mean_{p}'] = df['close'].rolling(p).mean() # Overlap: Rolling Mean
            if p != 1:
                df[f'rolling_std_{p}'] = df['close'].rolling(p).std() # Statistics Indicator: Rolling Standard Deviation
                df[f'zscore_{p}'] = (df['close'] - df[f'rolling_mean_{p}']) / df[f'rolling_std_{p}'] # Statistics Indicator: Z-score
    
        # Volume Indicator
        df['obv'] = ta.obv(df['close'], df['volume']) # OBV
        df['pvr'] = df['close_return_1'] * df['volume_change_1'] # PVR
        df['aobv'] = df['obv'].rolling(5).mean() # AOBV
    
        # Trend Indicator
        df['ttm_trend'] = np.where(df['ema_20_close'] > df['ema_50_close'], 1, 0) # TTM Trend
        
        df = pd.concat([
            df,
            ta.adx(df['high'], df['low'], df['close']), # Trend Indicator: ADX
            ta.cci(df['high'], df['low'], df['close']), # Momentum Indicator: CCI
            ta.cmo(df['close']), # Momentum Indicator: CMO
            ta.mfi(df['high'], df['low'], df['close'], df['volume']), # Momentum Indicator: MFI
            ta.roc(df['close']), # Momentum Indicator: ROC
            ta.trix(df['close']), # Momentum Indicator: TRIX
            ta.uo(df['high'], df['low'], df['close']), # Momentum Indicator: UO
            ta.wma(df['close']) # Overlap Indicator: WMA
        ], axis=1)
        
        # Candle Indicators
        df['inc'] = np.where(df['close'] > df['open'], 1, 0)
        df['dec'] = np.where(df['close'] < df['open'], 1, 0)
        df['cdl_doji'] = ta.cdl_doji(df['open'], df['high'], df['low'], df['close'])
        df['price_range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['range_ratio'] = df['body'] / (df['price_range'] + 1e-9)

        # Class Assignment
        df['class'] = np.where(df['open'].diff() > 0, 1, -1)
        df.dropna(inplace = True)

        df.to_csv(f'{self.path}/processed/{symbol}_indicators.csv', index = False)
    
        return df