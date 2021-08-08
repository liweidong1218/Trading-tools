#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import requests, json, os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr
import mplfinance as mpf


def get_all_price(tickers, start, end):
    try:
        tickers_s1 = tickers.tolist() # S&P 500 
    except:
        tickers_s1 = tickers
    ## Pull price
    yf.pdr_override() 
    price_df = pdr.get_data_yahoo(tickers_s1, start=start, end=end)

    ## Swap column levels
    try:
        price_df.columns = price_df.columns.swaplevel(0, 1)
        price_df.sort_index(axis=1, level=0, inplace=True)
        price_df.columns.set_levels(price_df.columns.levels[1].str.lower(),level=1,inplace=True)
    except:
        price_df.columns = price_df.columns.str.lower()
    
    return price_df

def compute_returns(df, ohlc_type):
    p = df[ohlc_type]
    returns = 100 * ((p.shift(-1) - p) / p).fillna(0.0)
    return returns.shift(1).fillna(0)

def plot(df):
    mpf.plot(df,type='candle')
    
def save_to_file(df, filename):
    mpf.plot(df,type='candle',savefig=filename)