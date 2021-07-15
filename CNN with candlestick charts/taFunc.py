#!/usr/bin/env python
# coding: utf-8
# Author: Liwei Dong

# In[ ]:

import os, json
import datetime as dt
import numpy as np
import pandas as pd
import ta

from ta.utils import dropna
from ta.volatility import BollingerBands

'''All input dataframe should have lower case column names'''

def get_sma(close, window_size, fill_na):
    '''ta.trend.SMAIndicator'''
    ta_sma = ta.trend.SMAIndicator(close, window = window_size, fillna = fill_na)
    return ta_sma.sma_indicator()
            
            
def get_macd(close, window_size_fast, window_size_slow, window_sign, fill_na):       
    '''ta.trend.MACD'''
    ta_macd = ta.trend.MACD(close, window_slow = window_size_slow, window_fast = window_size_fast, window_sign = window_sign, fillna = fill_na)
    return ta_macd.macd()


def get_stochastic_oscillator(df, window_size, smooth_window_size, fill_na):
    '''ta.momentum.StochasticOscillator'''
    ta_stochastic_kd = ta.momentum.StochasticOscillator(high = df.high, low = df.low, close = df.close, window = window_size, smooth_window = smooth_window_size, fillna = fill_na)
    stoch_k = ta_stochastic_kd.stoch() # stoch k
    stoch_d = ta_stochastic_kd.stoch_signal()# stoch d
    return (stoch_k, stoch_d)


def get_rsi(close, window_size, fill_na):
    '''ta.momentum.RSIIndicator'''
    ta_rsi = ta.momentum.RSIIndicator(close = close, window = window_size, fillna = fill_na)
    return ta_rsi.rsi()


def get_wr(df, lbp_, fill_na):
    '''ta.momentum.WilliamsRIndicator'''
    ta_wr = ta.momentum.WilliamsRIndicator(high = df.high, low = df.low, close = df.close, lbp = lbp_, fillna = fill_na)
    return ta_wr.williams_r()


def get_obv(close, volume, fill_na):
    '''ta.volume.OnBalanceVolumeIndicator'''
    ta_obv = ta.volume.OnBalanceVolumeIndicator(close = close, volume = volume, fillna = fill_na)
    return ta_obv


def get_mfi(df, window_size, fill_na):
    '''ta.volume.MFIIndicator'''
    ta_mfi = ta.volume.MFIIndicator(high = df.high, low = df.low, close = df.close, volume = df.volume, window = window_size, fillna = fill_na)
    return ta_mfi


def get_bb(close, window_size, dev_):
    # Initialize Bollinger Bands Indicator
    indicator_bb = BollingerBands(close = close, window = window_size, window_dev = dev_)

    # Add Bollinger Bands features
    bbm = indicator_bb.bollinger_mavg()
    bbh = indicator_bb.bollinger_hband()
    bbl = indicator_bb.bollinger_lband()

    # # Add Bollinger Band high indicator
    # df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

    # # Add Bollinger Band low indicator
    # df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()
    return bbm, bbh, bbl
