#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os, json, requests
import datetime as dt
import pandas as pd
import pymysql
import mysql.connector
import numpy as np
import smtplib

# Get all Expiry 
def getExpiry(opt): 
    exp_call = list(opt['callExpDateMap'].keys())
    exp_put = list(opt['putExpDateMap'].keys())
    if exp_call == exp_put:
        expiry_date = exp_call
    else:
        expiry_date = [value for value in exp_call if value in exp_put] 
    return expiry_date


# Get Strikes for single expiry
# style: 'Call' or 'Put'; exp_date: should be from the output of getExpiry (this needs to be optimized)
def getStrike(opt, style, exp_date): 
    if style == 'Call':
        strike = list(map(float, list(opt['callExpDateMap'][exp_date].keys())))
    if style == 'Put':
        strike = list(map(float, list(opt['putExpDateMap'][exp_date].keys())))
    return strike


# Get call / put dataframe for single expiry
def getChaindf(opt, exp_date, time_now):
    bid, ask, last, volume, open_interest, volatility, delta, gamma, theta, vega, rho = ([], ) * 11

    Map = opt['callExpDateMap'][exp_date]
    for strike in Map.keys():
        bid = bid + [Map[strike][0]['bid']]
        ask = ask + [Map[strike][0]['ask']]
        last = last + [Map[strike][0]['last']]
        volume = volume + [Map[strike][0]['totalVolume']]
        open_interest = open_interest + [Map[strike][0]['openInterest']]
        volatility = volatility + [Map[strike][0]['volatility']]
        delta = delta + [Map[strike][0]['delta']]
        gamma = gamma + [Map[strike][0]['gamma']]
        theta = theta + [Map[strike][0]['theta']]
        vega = vega + [Map[strike][0]['vega']]
        rho = rho + [Map[strike][0]['rho']]

    call_dict = {'strike':list(map(float, list(Map.keys()))),'bid_c':bid, 'ask_c': ask, 'last_c':last, 'volume_c':volume, 'open_interest_c':open_interest, 'volatility_c':volatility, 'delta_c':delta, 'gamma_c':gamma, 'theta_c':theta, 'vega_c':vega, 'rho_c':rho}
    call_df = pd.DataFrame(call_dict)
    call_df['Expiry'] = exp_date.split(':')[0]
    call_df['Date'] = time_now
    
    bid, ask, last, volume, open_interest, volatility, delta, gamma, theta, vega, rho = ([], ) * 11
    
    Map = opt['putExpDateMap'][exp_date]
    for strike in Map.keys():
        bid = bid + [Map[strike][0]['bid']]
        ask = ask + [Map[strike][0]['ask']]
        last = last + [Map[strike][0]['last']]
        volume = volume + [Map[strike][0]['totalVolume']]
        open_interest = open_interest + [Map[strike][0]['openInterest']]
        volatility = volatility + [Map[strike][0]['volatility']]
        delta = delta + [Map[strike][0]['delta']]
        gamma = gamma + [Map[strike][0]['gamma']]
        theta = theta + [Map[strike][0]['theta']]
        vega = vega + [Map[strike][0]['vega']]
        rho = rho + [Map[strike][0]['rho']]

    put_dict = {'strike':list(map(float, list(Map.keys()))),'bid_p':bid, 'ask_p': ask, 'last_p':last, 'volume_p':volume, 'open_interest_p':open_interest, 'volatility_p':volatility, 'delta_p':delta, 'gamma_p':gamma, 'theta_p':theta, 'vega_p':vega, 'rho_p':rho}
    put_df = pd.DataFrame(put_dict)
    put_df['Expiry'] = exp_date.split(':')[0]
    put_df['Date'] = time_now
    
    return (call_df, put_df)

def minute_chain(ticker, opt,filename):
    if opt['status'] == 'FAILED':
        return pd.DataFrame()
    else:
        col = ['putCall','bid', 'ask', 'last', 'totalVolume', 'openInterest',
       'volatility', 'delta', 'gamma', 'theta', 'vega', 'rho']
        cpflag = ['callExpDateMap','putExpDateMap']
        df = pd.DataFrame.from_dict({(i,j,k): opt[i][j][k][0]
                                    for i in cpflag
                                    for j in opt[i].keys() 
                                    for k in opt[i][j].keys()},
                               orient='index',columns=col)
        
        time_now = dt.datetime.strptime(filename[len(ticker)+1:-8],'%b %d %H_%M_%S %Y') 
        df['Date'] = time_now
        return df


# Get put/call ratio for single expiry
def getRatio(opt, exp_date):
    call_volume, put_volume = ([], ) * 2

    Map = opt['callExpDateMap'][exp_date]
    for strike in Map.keys():
        call_volume = call_volume + [Map[strike][0]['totalVolume']]
    
    Map = opt['putExpDateMap'][exp_date]
    for strike in Map.keys():
        put_volume = put_volume + [Map[strike][0]['totalVolume']]

    ratio = sum(put_volume) / sum(call_volume)
    return ratio


def get_option_chain(**kwargs):
    key = 'UPGQCPR4GR9EGFDMUADAM0N8ICKGVMXO'
    url = 'https://api.tdameritrade.com/v1/marketdata/chains?&symbol={}'.format(kwargs.get('symbol'))

    params = {}
    params.update({'apikey': key})

    for arg in kwargs:
        parameter = {arg: kwargs.get(arg)}
        params.update(parameter)
    
    params.pop('symbol')
    
    return requests.get(url = url, params=params).json()


def get_price_history(**kwargs):
    key = 'UPGQCPR4GR9EGFDMUADAM0N8ICKGVMXO'
    
    url = 'https://api.tdameritrade.com/v1/marketdata/{}/pricehistory'.format(kwargs.get('symbol'))

    params = {}
    params.update({'apikey': key})

    for arg in kwargs:
        parameter = {arg: kwargs.get(arg)}
        params.update(parameter)
    
    return requests.get(url, params=params).json()







###### MySQL database functions ######

def checkTableExists(dbcon, tablename):
    dbcur = dbcon.cursor()
    dbcur.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = '{0}'
        """.format(tablename.replace('\'', '\'\'')))
    if dbcur.fetchone()[0] == 1:
        dbcur.close()
        return True

    dbcur.close()
    return False

def appendDFToCSV_void(df, csvFilePath, opt_type, sep=","):
    
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='w', index=False, sep=sep)
    else: 
        if not df.Date.iloc[0] == pd.read_csv(csvFilePath, sep=',').Date.iloc[-1]:
            if len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
                raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
            else: 
                df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)

            if not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
                raise Exception("Columns and column order of dataframe and csv file do not match!!")
            else:
                df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)
        else: 
            print(opt_type + ": File already up-to-date!")
            

def sendEmail(sender, password, receiver, subject, message):
    gmail_user = sender
    gmail_pwd = password
    FROM = sender
    TO = receiver if type(receiver) is list else [receiver]
    SUBJECT = subject
    TEXT = message

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print ('successfully sent the mail')
    except:
        print ('failed to send mail')






###### Option Chain Analysis ######

def readChain(ticker, folder_path, date_list): 
    '''
    read txt into json and then output pandas dataframe 
    inputs:
        - ticker: str
        - folder_path: str
        - dates: list of datetime.date()
    pending item: 
        - input: include dates variable to select certain date / time for analysis purpose
    '''
    dir_list = os.listdir(folder_path)
    call_df_list = []
    put_df_list = []
    
    if not date_list:
        '''read in all dates in database'''
        for filename in dir_list: ###  speed up
            with open(folder_path + '/' + filename, 'r') as read_file:
                opt = json.load(read_file) # speed up

            expiry_date = getExpiry(opt)
            time_now = dt.datetime.strptime(filename[len(ticker)+1:-4],'%a %b %d %H_%M_%S %Y') 

            t = 0
            for exp_date in expiry_date: # speed up
                t = t + 1
                if t == 1:
                    output_tup = getChaindf(opt, exp_date, time_now)
                    call_df = output_tup[0]
                    put_df = output_tup[1]
                else:
                    output_tup = getChaindf(opt, exp_date, time_now)
                    call_df = call_df.append(output_tup[0])
                    put_df = put_df.append(output_tup[1])

            call_df_list.append(call_df)
            put_df_list.append(put_df)

        call_raw_df = pd.concat(call_df_list)
        call_raw_df.reset_index(drop = True, inplace = True)
        put_raw_df = pd.concat(put_df_list)
        put_raw_df.reset_index(drop = True, inplace = True)
    
    else:
        '''read in dates specified by the user'''
        file_list = [x for x in dir_list if dt.datetime.strptime(x[len(ticker)+1:-4],'%a %b %d %H_%M_%S %Y').date() in date_list]
        
        for filename in file_list:
            with open(folder_path + '/' + filename, 'r') as read_file:
                opt = json.load(read_file)

            expiry_date = getExpiry(opt)
            time_now = dt.datetime.strptime(filename[len(ticker)+1:-4],'%a %b %d %H_%M_%S %Y') 

            t = 0
            for exp_date in expiry_date:
                t = t + 1
                if t == 1:
                    output_tup = getChaindf(opt, exp_date, time_now)
                    call_df = output_tup[0]
                    put_df = output_tup[1]
                else:
                    output_tup = getChaindf(opt, exp_date, time_now)
                    call_df = call_df.append(output_tup[0])
                    put_df = put_df.append(output_tup[1])

            call_df_list.append(call_df)
            put_df_list.append(put_df)

        call_raw_df = pd.concat(call_df_list)
        call_raw_df.reset_index(drop = True, inplace = True)
        put_raw_df = pd.concat(put_df_list)
        put_raw_df.reset_index(drop = True, inplace = True)        
        
    
    return (call_raw_df, put_raw_df)


def day_chain(ticker, folder_path, date_list):
    dir_list = os.listdir(folder_path)
    file_list = [x for x in dir_list if dt.datetime.strptime(x[len(ticker)+1:-8],'%b %d %H_%M_%S %Y').date() in date_list]
    df = []
    
    for filename in file_list:
        with open(folder_path + '/' + filename, 'r') as read_file:
            opt = json.load(read_file)
        df.append(minute_chain(ticker,opt,filename))
        
    df = pd.concat(df)  
    df.index.names = ['cpflag','Expiry','Strike']
    df.reset_index(inplace=True)
    df.drop(columns='cpflag',inplace=True)
    df['Expiry'] = df['Expiry'].apply(lambda x:x.split(':')[0])
    df.set_index('Date',inplace=True)
    
    return df


def rankExpiry(df, method):
    vol_rank = df.groupby(['Expiry']).sum().volume_c.rank(method = method)
    oi_rank = df.groupby(['Expiry']).sum().open_interest_c.rank(method = method)
    expiry_score = (vol_rank + oi_rank).sort_values(ascending = False)
    return (expiry_score.index[0], expiry_score)


def rankStrike(df, method):
    vol_rank = df.groupby(['strike']).sum().volume_c.rank(method = method)
    oi_rank = df.groupby(['strike']).sum().open_interest_c.rank(method = method)
    strike_score = (vol_rank + oi_rank).sort_values(ascending = False)
    return (strike_score.index[0], strike_score)





            
            
        
