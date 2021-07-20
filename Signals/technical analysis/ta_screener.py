
### stock screener based on technical indicators & patterns ###
import os, sys
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import ta
from workers import *

project_dir = os.path.realpath(os.path.join(os.getcwd(), '..','..','..'))
tool_dir = os.path.join(project_dir, 'Utils')
sys.path.append(tool_dir)


class technicalIndicator:
    def __init__(self, ticker, start_date, end_date):

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.df = get_all_price(self.ticker, self.start_date, self.end_date)

        self.columns = {'open','high','low','close','volume'}
        assert self.columns.issubset(self.df.columns)
        
        self.df = self.df
        self.open = self.df.open
        self.high = self.df.high
        self.low = self.df.low
        self.close = self.df.close
        self.volume = self.df.volume
        self.atr()
        self.rsi()
        
    def run(self, strategy='reversal'):
        if strategy == 'reversal':
            index = self.reversal_functions()
            return self.df[index]
    
    def reversal_functions(self):
        return self.close_minus_open_filter() & \
               self.open_minus_low_filter() & \
               self.volume_filter() & \
               self.neg_cumret_filter() & \
               self.rsi_filter()
                
    #filters
    def close_minus_open_filter(self, k=0.5, weak_signal = False):
        """
        close - open >= K*atr
        """
        if weak_signal == False:
            index = self.close - self.open >= k*self.atr
        else:
            index = abs(self.close - self.open) <= k*self.atr
        
        return index
    
    def open_minus_low_filter(self, k=0.5):
        """
        Open - Low >= K*(Close - Open)
        """
        index = self.open - self.low >= k*(self.close-self.open)
        return index

    def volume_filter(self, periods=5):
        """
        Current volume > volume MA 
        """
        lag_volume = self.volume.rolling(periods).mean().shift()
        index = self.volume > lag_volume
        return index

    def neg_cumret_filter(self, periods=5):
        """
        Cumulative return is negative for previous n days
        """
        lag_close = self.close.shift(periods)
        index = self.close < lag_close
        return index
    
    def rsi_filter(self, periods=5, threshold = None):
        """
        Current RSI < min(RSI in the past n days)
        """
        if threshold is None:
            rolling_rsi = self.rsi.rolling(periods).min().shift()
            index = self.rsi < rolling_rsi
        else:
            index = self.rsi <= threshold
        return index
        
    #technical indicators
    def atr(self, n=14):
        data = self.df.copy()
        data['tr0'] = abs(self.high - self.low)
        data['tr1'] = abs(self.high - self.close.shift())
        data['tr2'] = abs(self.low - self.close.shift())
        tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
        self.atr = self.wwma(tr, n)
    
    def wwma(self, values, n):
        """
        J. Welles Wilder's EMA 
        """
        return values.ewm(alpha=1/n, adjust=False).mean()
    
    def rsi(self):
        self.rsi = ta.momentum.rsi(self.close, window=14)
    
    @classmethod
    def plot_filtered_date(cls, full_df, filtered_date):
        assert 'close' in full_df.columns, 'close column missing'
        close = full_df.close
        plt.figure(figsize=(10,4))
        plt.plot(close)
        plt.scatter(filtered_date, close.loc[filtered_date], c='r')


class quick_screener:
    '''Works for multiple tickers'''
    # constructor
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = get_all_price(self.ticker, self.start_date, self.end_date)

    def show_name(self):
        print("Tickers are: ")
        print(*self.ticker, sep = ", ")
        
    def print_data(self):
        self.data.head()
    
    def sma_hit(self, ma_window, threshold):
        # default is to use close price
        df = self.data.xs('close', axis = 1, level = 1, drop_level=True)
        
        sma_df = df.apply(get_sma, axis = 0, window_size = ma_window, fill_na = True)
        diff = df.iloc[-1] - sma_df.iloc[-1]
        pct_diff = diff / df.iloc[-1]
        return pct_diff.loc[(pct_diff>-threshold)&(pct_diff<threshold)]
    
    def HH_breakout(self, window, min_period, past_days):
        '''Detect if stock stand above x-period MA'''
        # default is to use close price
        # need to update to find the first time stand above x-period MA
        # current version shows a small universe around x-period MA
        df = self.data.xs('close', axis = 1, level = 1, drop_level=True)
        
        max_list_all = df.rolling(window = window, min_periods = min_period).max()
        max_price = max_list_all.max()
        current_price = df.iloc[-1]
        max_idx = max_list_all.idxmax(axis = 0)
        current_idx = max_list_all.index[-1]
        
        days = (max_idx - current_idx).dt.days
        slope = (max_price - current_price) / (max_idx - current_idx).dt.days
        
        brk_df = pd.concat([slope, days], axis = 1).sort_values(by = 0, ascending = False)
        brk_df.dropna(inplace = True)
        brk_df.columns = ['Slope', 'Days']
        
        return brk_df.loc[brk_df.Days > -past_days]

    def volume_surge(self, method, window, min_period, decay, threshold, past_days):
        '''Detect stock if current volume > a multiple of MA volume'''
        df = self.data.xs('volume', axis = 1, level = 1, drop_level=True)
        
        if method == 'SMA':
            MAvol = df.rolling(window, min_periods=1).mean()
        else:
            MAvol = df.ewm(com = decay, ignore_na = True).mean()
        
        return df.columns[(df.iloc[-past_days:, :] >= threshold*MAvol.iloc[-1, :]).any()].tolist()
    
    def sma_cross(self):
        '''Detect stock when SMA cross'''
        pass
    
    def consolidation_breakout():
        '''Detect stock that has been consolidated for certain period'''
        pass