import os
import sys

repository_dir = os.path.realpath(os.path.join(os.getcwd(), '..','..','..'))
util_dir = os.path.join(repository_dir, 'utils')
sys.path.append(util_dir)

from workers import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
plt.style.use('seaborn')


def adjust_stock_split(full_df, stock_split_date=None, split_ratio=None):
    output = full_df.copy()
    quotedate = output['quotedate']
    
    if stock_split_date is not None:
        stock_split_date = pd.to_datetime(stock_split_date)
        output.loc[quotedate < stock_split_date, 'underlying_last'] /= split_ratio
        output.loc[quotedate < stock_split_date, 'volume'] *= split_ratio
    return output

def train_test_split(full_df, split_date, window):
    dates = full_df['quotedate'].unique()
    assert len(dates) > window, 'window too large'
    
    split_date = pd.to_datetime(split_date)
    i = 0
    while split_date not in dates:
        split_date -= pd.Timedelta(days=1)
        i += 1
        if i > 10:
            break

    split_date_index = np.argwhere(dates == split_date)
    if split_date_index.size == 0:
        test_date_index = 0
    else:
        test_date_index = split_date_index[0][0] - window
    test_start_date = dates[test_date_index]
    
    train = full_df[full_df['quotedate'] < split_date]
    test = full_df[full_df['quotedate'] >= test_start_date]
    return train, test


def quotedate_contract(df, start, end, splitdate=None, splitratio=None):
    """
    get contracts with quotedate within start and end
    """
    output = df.copy()
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    quotedate = output['quotedate']
    
    if splitdate is not None:
        splitdate = pd.to_datetime(splitdate)
        output.loc[quotedate < splitdate, 'underlying_last'] /= splitratio
        output.loc[quotedate < splitdate, 'volume'] *= splitratio
    
    index = (quotedate >= start) & (quotedate <= end)
    
    return output[index]
    
def monthly_contract(df):
    """
    return 3 types of monthly contract: expire in current/next/3rd month
    """
    expiration, quotedate = df['expiration'], df['quotedate']
    expr_month = expiration.dt.year * 12 + expiration.dt.month
    quote_month = quotedate.dt.year * 12 + quotedate.dt.month
    
    third_friday = (expiration.dt.day >=15) & (expiration.dt.day <= 21) & (expiration.dt.weekday == 4)
    cur_month_expr =  expr_month == quote_month
    next_month_expr = expr_month - quote_month == 1
    third_month_expr = expr_month - quote_month == 2

    return df[third_friday&cur_month_expr], df[third_friday&next_month_expr], df[third_friday&third_month_expr]

def integer_strike_contract(df):
    """
    return contract whose strike is multiple of 10
    """
    strike = df['strike']
    int_strike = strike % 10 == 0

    return df[int_strike]

def moneyness_contract(df, atm_delta=[0.4,0.6], otm_delta=0.4, itm_delta=0.6):
    delta = df['delta']
    atm = df[(abs(delta) >= atm_delta[0]) & (abs(delta) <= atm_delta[1])]
    otm = df[abs(delta) < otm_delta]
    itm = df[abs(delta) > itm_delta]
    
    return atm, otm, itm
    
def pcratio(df):
    """
    return put-call ratio by date
    """
    ratio = df[df['type']=='put'].groupby('quotedate')['volume'].sum() / df[df['type']=='call'].groupby('quotedate')['volume'].sum()
    return ratio
    
def plot_by_moneyness(df, atm_delta=[0.4,0.6], otm_delta=0.4, itm_delta=0.6, _type='call', 
                      groupby='quotedate', variable = ['impliedvol','volume','openinterest'], style='scatter',title='None'):
    """
    plot n*3 graphs, n variables * 3 moneyness
    """
    atm, otm, itm = moneyness_contract(df, atm_delta, otm_delta, itm_delta)
    
    op = {'impliedvol':'mean', 'volume':'max', 'openinterest':'mean','underlying_last':'mean','delta':'mean'}
    grouped0 = atm[atm['type']==_type].groupby(groupby).agg(op)
    grouped1 = otm[otm['type']==_type].groupby(groupby).agg(op)
    grouped2 = itm[itm['type']==_type].groupby(groupby).agg(op)
    if 'pcratio' in variable:
        grouped0['pcratio'] = pcratio(atm)
        grouped1['pcratio'] = pcratio(otm)
        grouped2['pcratio'] = pcratio(itm)
    
    row, col = len(variable), 3
    fig = plt.figure(figsize=(24,row*6)) 
    fig.suptitle(title, fontsize=14,y=0.95)
    for r in range(row):
        for c in range(col):
            i = r*col + c + 1
            ax = fig.add_subplot(row, col, i)
            if c==0:
                y = grouped0[variable[r]]
                x = grouped0.index
                if style == 'scatter':
                    a = ax.scatter(x,y,c=grouped0['delta'],marker='+',cmap='viridis')
                    fig.colorbar(a, ax=ax, pad=0.1)
                if style == 'ts':
                    ax.plot(x,y,c='y')
                if groupby=='quotedate':
                    ax1 = ax.twinx()
                    ax1.plot(grouped0['underlying_last'], c='r')
                ax.set_title(variable[r]+' atm')
                ax.set(xlabel=groupby, ylabel=variable[r])
                
            if c==1:
                y = grouped1[variable[r]]
                x = grouped1[variable[r]].index
                if style == 'scatter':
                    a = ax.scatter(x,y,c=grouped1['delta'],alpha=0.5,cmap='viridis')
                    fig.colorbar(a, ax=ax, pad=0.1)
                if style == 'ts':
                    ax.plot(x,y)
                if groupby=='quotedate':
                    ax1 = ax.twinx()
                    ax1.plot(grouped1['underlying_last'], c='r')
                ax.set_title(variable[r]+' otm')
                ax.set(xlabel=groupby, ylabel=variable[r])
                
            if c==2:
                y = grouped2[variable[r]]
                x = grouped2[variable[r]].index
                if style == 'scatter':
                    a = ax.scatter(x,y,c=grouped2['delta'],marker='x',cmap='viridis')
                    fig.colorbar(a, ax=ax, pad=0.1)
                if style == 'ts':
                    ax.plot(x,y,c='g')
                if groupby=='quotedate':
                    ax1 = ax.twinx()
                    ax1.plot(grouped2['underlying_last'], c='r')
                ax.set_title(variable[r]+' itm')
                ax.set(xlabel=groupby, ylabel=variable[r])
    return atm, otm, itm


def anomaly_dates(df, _type='call', variable=['volume','pcratio'], direction=['>','<'], 
                  thresh=[3,0.5], window=30, volume_operation='max', log_trans=False):
    """
    return dates in which variables exceed thresh
    """
    grouped = df[df['type']==_type].groupby('quotedate')
    conditions = []
    for i, v in enumerate(variable):
        if v=='volume':
            vol_series = grouped['volume'].agg(volume_operation)
            if log_trans: vol_series = np.log(vol_series)
            vol_rollingmean = vol_series.rolling(window).mean().shift()
            vol_rollingstd = vol_series.rolling(window).std().shift()
            vol_thresh = vol_rollingmean + vol_rollingstd*thresh[i]
            cond = pd.eval('vol_series' + direction[i] + 'vol_thresh') 
            
        if v=='pcratio':
            pcr = pcratio(df)
            cond = pd.eval('pcr' + direction[i] + str(thresh[i])) 
        
        if v=='openinterest':
            oi_series = grouped['openinterest'].mean()
            if log_trans: oi_series = np.log(oi_series)
            oi_rollingmean = oi_series.rolling(window,min_periods=1).mean().shift()
            oi_rollingstd = oi_series.rolling(window,min_periods=1).std().shift()
            oi_thresh = oi_rollingmean + oi_rollingstd*thresh[i]
            cond = pd.eval('oi_series' + direction[i] + 'oi_thresh')
            
        if v=='impliedvol':
            iv_series = grouped['impliedvol'].mean()
            iv_rolling = iv_series.rolling(window).quantile(thresh[i], interpolation='linear').shift()
            cond = pd.eval('iv_series' + direction[i] + 'iv_rolling')
        conditions.append(cond)
        
    dates = pd.concat(conditions, axis=1).all(axis=1)
    dates = dates[dates==1].index
    return dates

def outlier_impact(optiondata, outlier_date, win_thresh=0, verbose=True):
    price_series = optiondata.groupby('quotedate')['underlying_last'].mean()
    
    outlier_date_index = np.argwhere(np.in1d(price_series.index,outlier_date)).reshape(-1)  
    next_day_index = np.clip(outlier_date_index + 1, 0, len(price_series)-1)
    next_5day_index = np.clip(outlier_date_index + 5, 0, len(price_series)-1)

    a = price_series[outlier_date_index].reset_index(drop=True)
    b = price_series[next_day_index].reset_index(drop=True)
    c = price_series[next_5day_index].reset_index(drop=True)
    
    output = pd.concat([a,b,c],axis=1)
    output.index = outlier_date
    output.columns = ['outlierday','nextday','next5thday']
    
    next_day_return = np.log(b/a) 
    next_5thday_return = np.log(c/a)
    #avg_5thday_winprob = round((c>a).mean(),3)
    avg_5thday_winprob = round((next_5thday_return>win_thresh).mean(),3)
    avg_5thday_posreturn = round(next_5thday_return[next_5thday_return>0].mean(),3)
    avg_5thday_negreturn = round(next_5thday_return[next_5thday_return<0].mean(),3)
    avg_5thday_expreturn = avg_5thday_posreturn*avg_5thday_winprob + avg_5thday_negreturn*(1-avg_5thday_winprob)
    
    if verbose:
        print('+1 day stock price increase probability:', round((b>a).mean(),3))
        print('+1 day stock price avg increase:', round(next_day_return[next_day_return>0].mean(),3))
        print('+1 day stock price avg decrease', round(next_day_return[next_day_return<0].mean(),3),'\n')
        print('+5 day stock price increase probability:', avg_5thday_winprob)
        print('+5 day stock price avg increase:', avg_5thday_posreturn)
        print('+5 day stock price avg decrease:', avg_5thday_negreturn)
        print('+5 day expected return:', avg_5thday_expreturn)
        return output
    
    else:
        return avg_5thday_winprob, len(output), avg_5thday_posreturn, avg_5thday_negreturn, avg_5thday_expreturn

def optimize_threshold(df, _type='call', variable=['volume','pcratio','impliedvol'], direction=['>','<','>'], 
                       volume_operation='max', window=252, win_thresh=0, log_trans=False, **kwargs):
    
    thresh_dic = {'volume':np.arange(1,3.1,0.1), 'pcratio':np.arange(0.5,1.6,0.1),
                  'impliedvol':np.arange(0,1,0.1), 'openinterest':np.arange(1,3.1,0.1)}
    thresh_dic = {k:thresh_dic[k] for k in variable}
    for key, value in kwargs.items():
        assert key in thresh_dic, f'{key} is invalid variable'
        thresh_dic[key] = value
            
    l = []
    for i, param in enumerate(itertools.product(*thresh_dic.values())):
        outlier = anomaly_dates(df, _type=_type, variable=variable, direction=direction, 
                                thresh=param, window=window, volume_operation=volume_operation, log_trans=log_trans)
        stats = outlier_impact(df, outlier, win_thresh=win_thresh, verbose=False)
        l.append([*stats,np.around(param,2)])
        if i%500 == 0:
            print(f'Testing {i}th combo')
            
    performance = pd.DataFrame(l,columns=['winprob','outliernum','posret','negret','expret','/'.join(variable)])
    return performance

def filter_threshold(performance, winprob=0.7, topn=1):
    #get top 3 winning probability entries in each group
    temp = performance[(performance['outliernum']>1) & (performance['winprob']>winprob)]
    temp = temp.sort_values(['outliernum','winprob'],ascending=[False,False]).groupby('outliernum').head(topn)
    return temp

def highest_volume_strike(df, date, _type='call', splitdate=None, splitratio=None):
    """
    get strike that has most volume on date
    """
    date, splitdate = pd.to_datetime(date), pd.to_datetime(splitdate)
    strike = df[(df['quotedate']==date) & (df['type']==_type)].sort_values('volume',ascending=False).iloc[0]['strike']
    if (splitdate is not None) and (date < splitdate):
        strike /= splitratio
        
    return strike

def snapshot(df, date, ticker, before=20, after=10):
    """
    plot stock price series before and after the date
    """
    start = pd.to_datetime(date) - np.timedelta64(before,'D')
    end = pd.to_datetime(date) + np.timedelta64(after,'D')
    close = get_all_price(ticker, start, end)['close']
    highest_volume_K = highest_volume_strike(df, date, _type='call', splitdate='2020-8-31', splitratio=5)
    
    fig = plt.figure()
    plt.plot(close.T)
    plt.hlines(highest_volume_K,start,end,'y',label=str(highest_volume_K))
    plt.title(date)
    plt.scatter(date,close.loc[date],c='r',marker='x')
    
def winrate(stock_series, win_thresh=-0.01, lag=5):
    lag_series = stock_series.shift(lag)
    ret = np.log(stock_series / lag_series)
    win_rate = np.mean(ret>win_thresh)
    
    return round(win_rate,3)