import pandas as pd
import gc, os
from zipfile import  ZipFile #pip install zipfile36

#### read from parquet ### 
def generate_file_list(folderpath, start, end):
    start += '01'
    end += '01'
    month_list = pd.date_range(start, end, freq='MS').strftime('%Y%m')
    #folderPath + date[:4] + f'/ORATS_SMV_Strikes_{date}.zip'
    
    file_list = [os.path.join(folderpath, m[:-2], 
                              f'ORATS_SMV_Stries_{m}.parquet') for m in month_list]
    return file_list
    
def cleanDF(df, ticker=None):
    callCol = ['ticker','stkPx','expirDate','strike','cValue','cBidPx', 'cAskPx', 'cVolu', 'cOi',
               'cMidIv', 'delta', 'gamma', 'theta', 'vega','quotedate']
    putCol = ['ticker','stkPx', 'expirDate','strike','pValue','pBidPx', 'pAskPx', 'pVolu', 'pOi',
               'pMidIv', 'delta', 'gamma', 'theta', 'vega','quotedate']
    commonCol = ['underlying','underlying_last', 'expiration', 'strike', 'last', 'bid', 'ask', 'volume', 
                 'openinterest', 'impliedvol', 'delta', 'gamma', 'theta', 'vega','quotedate', 'type', 'dte', 'optionroot']
    
    if ticker is not None:
        df = df.loc[df['ticker'].isin(ticker)].copy()
        
    df['quotedate'] = df.index
    
    call = df[callCol].copy()
    call['type'] = 'call'
    call['dte'] = (call['expirDate'] - call['quotedate']).dt.days
    call['optionroot'] = call['ticker'] + call['expirDate'].dt.strftime('%d%m%y') + 'C0' + call['strike'].astype(int).astype(str)
    
    put = df[putCol].copy()
    put['type'] = 'put'
    put['dte'] = (put['expirDate'] - put['quotedate']).dt.days
    put['optionroot'] = put['ticker'] + put['expirDate'].dt.strftime('%d%m%y') + 'P0' + put['strike'].astype(int).astype(str)
    
    call.columns = commonCol
    put.columns = commonCol
    
    output_df = pd.concat([call,put])
    output_df.reset_index(inplace = True, drop = True)
    
    data = output_df[['underlying', 'underlying_last', 'optionroot', 'type', 'expiration', 'quotedate', 'strike', 'last', 'bid', 'ask', 
             'volume', 'openinterest', 'impliedvol', 'delta', 'gamma', 'theta', 'vega', 'dte']]
    
    return data


def cleanParquet(file, ticker=None):
    try:
        option_df = pd.read_parquet(file)
        cleaned_df = cleanDF(option_df, ticker)
        print(f'{file} done!')
        return cleaned_df
        
    except:
        print(f'{file} not found')
      
def read_optiondata(file_list, ticker=None):
    optionsdata = []
    for file in file_list:
        data = cleanParquet(file, ticker)
        optionsdata.append(data)
        del data
    optionsdata = pd.concat(optionsdata)
    return optionsdata
 
def read_optiondata_by_ticker(file_list, ticker):
    optionsdata = []
    for file in file_list:
        try:
            data = pd.read_parquet(file)
            data = data[data['ticker'].isin(ticker)]
            optionsdata.append(data)
            del data
            gc.collect()
            print(f'{file} done!')
        except:
            print(f'{file} not found')
            continue
            
    output = pd.concat(optionsdata)
    if output.empty:
        print('---not available---')
    else:
        output = cleanDF(output)
    return output


#### read from zip csv ### 

def readDayCsv(folderPath, date):
    '''
        folderPath: path to the zipfiles;
        date: '20160104', str
    '''
    zFile = folderPath + date[:4] + f'\\ORATS_SMV_Strikes_{date}.zip'
    csvFile = f'ORATS_SMV_Strikes_{date}.csv'
    with ZipFile(zFile) as z:
        with z.open(csvFile) as f:
            df = pd.read_csv(f, parse_dates = ['trade_date','expirDate'], index_col='trade_date')
    return df
    

def readDayOption(df, ticker=None):
    '''
        df: ORATS.csv read in as df
        output: call,put df with formatted columns
    '''
    callCol = ['ticker','stkPx','expirDate','strike','cBidPx', 'cAskPx', 'cVolu', 'cOi',
        'smoothSmvVol', 'delta', 'gamma', 'theta', 'vega', 'rho']
    putCol = ['ticker','stkPx', 'expirDate','strike','pBidPx', 'pAskPx', 'pVolu', 'pOi',
            'smoothSmvVol', 'delta', 'gamma', 'theta', 'vega', 'rho']
    commonCol = ['ticker','underlying', 'expiry', 'strike', 'bid', 'ask', 'totalVolume', 'openInterest',
            'volatility', 'delta', 'gamma', 'theta', 'vega', 'rho', 'putCall']

    call = df[callCol].copy()
    call['putCall'] = 'CALL'
    put = df[putCol].copy()
    put['putCall'] = 'PUT'
    call.columns = commonCol
    put.columns = commonCol
    if ticker is not None:
        call = call[call['ticker'].isin(ticker)]
        put = put[put['ticker'].isin(ticker)]

    return pd.concat([call,put])

def readMultiDay(folderPath, startDate, endDate, ticker=None):
    '''
        startDate, endDate: 20160104, 20160531, str
    '''
    dateList = [d.strftime('%Y%m%d') for d in pd.date_range(startDate, endDate)]
    df = []
    for d in dateList:
        try:
            # dayDf = readDayOption(readDayCsv(folderPath, d), ticker)
            dayDf = cleanDF(readDayCsv(folderPath, d), ticker)
            df.append(dayDf)
            print(f'{d} done!')
        except:
            print(f'{d} is not a trading day.')

    return pd.concat(df)