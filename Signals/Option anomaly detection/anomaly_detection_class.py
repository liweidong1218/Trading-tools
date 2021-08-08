import numpy as np
import pandas as pd
import itertools


class anomalyDetector:
    def __init__(self):
        self._atm_delta = [0.4,0.6]
        self._otm_delta = 0.4
        self._itm_delta = 0.6

        self._params = None
        self._thresh_dic =None
        self._train_contract = None
        self._train_outlier = None
        self._test_contract = None
        self._test_outlier = None
        self._train_performance = None
        self._test_performance = None

    def compile(self, _type=None, expiry=None, monthly=None, moneyness=None, params=None):
        self._type = _type
        self._expiry = expiry
        self._monthly = monthly
        self._moneyness = moneyness
        self.set_params(params)

    def set_params(self, params):
        self._thresh_dic = {'volume':np.arange(1,3.1,0.1), 'pcratio':np.arange(0.5,1.6,0.1),
                            'impliedvol':np.arange(0,1,0.1), 'openinterest':np.arange(1,3.1,0.1)}
        assert set(params['variable']).issubset(set(self._thresh_dic.keys())), 'invalid variable name'
        self._params = params
        self._thresh_dic = {k:self._thresh_dic[k] for k in params['variable']}

    def fit(self, train, win_thresh):
        self._train_contract = self.signal_contract(train)
        self._train_performance = self.optimize_threshold(win_thresh)
    
    def predict(self, test, win_thresh, param_thresh, verbose):
        self._test_contract = self.signal_contract(test)
        self._test_outlier = self.anomaly_dates(self._test_contract, param_thresh)
        test_perform = self.outlier_impact(self._test_contract, self._test_outlier, win_thresh=win_thresh, verbose=False)
        self._test_performance = self.record_to_df([[*test_perform, param_thresh]])
        if verbose:
            return self.outlier_impact(self._test_contract, self._test_outlier, win_thresh=win_thresh, verbose=verbose)
            

    def evaluate(self, winprob=0.7, topn=1):
        #get top 1 winning probability entries in each group
        performance = self._train_performance
        temp = performance[(performance['outliernum']>1) & (performance['winprob']>winprob)]
        temp = temp.sort_values(['outliernum','winprob'],ascending=[False,False]).groupby('outliernum').head(topn)
        return temp
    
    def signal_contract(self, df):
        """
        filter out contracts that act as outlier detectors
        """
        df = self.preprocess(df)
        cond = self.type_contract(df) & self.expiry_contract(df) & self.monthly_contract(df) & self.moneyness_contract(df)
        return df[cond]

    def anomaly_dates(self, df, thresh):
        """
        return dates in which variables exceed threshold
        """
        grouped = df.groupby('quotedate')
        conditions = []
        variable = self._params['variable']
        direction = self._params['direction']
        window = self._params['window']
        volume_operation = self._params['volume_operation']
        log_trans = self._params['log_trans']
        
        for i, v in enumerate(variable):
            if v=='volume':
                vol_series = grouped['volume'].agg(volume_operation)
                vol_series[vol_series==0] = 0.01
                if log_trans: vol_series = np.log(vol_series)
                vol_rollingmean = vol_series.rolling(window).mean().shift()
                vol_rollingstd = vol_series.rolling(window).std().shift()
                vol_thresh = vol_rollingmean + vol_rollingstd*thresh[i]
                cond = pd.eval('vol_series' + direction[i] + 'vol_thresh') 
                
            if v=='pcratio':
                pcr = grouped['pcratio'].first()
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
        
    def outlier_impact(self, df, outlier_date, win_thresh, verbose=True):
        price_series = df.groupby('quotedate')['underlying_last'].first()

        outlier_date_index = np.argwhere(np.in1d(price_series.index,outlier_date)).reshape(-1)  
        next_day_index = np.clip(outlier_date_index + 1, 0, len(price_series)-1)
        next_5day_index = np.clip(outlier_date_index + 5, 0, len(price_series)-1)

        a = price_series[outlier_date_index].reset_index(drop=True)
        b = price_series[next_day_index].reset_index(drop=True)
        c = price_series[next_5day_index].reset_index(drop=True)
        
        next_day_return = np.log(b/a) 
        next_5thday_return = np.log(c/a)
        avg_5thday_winprob = round((next_5thday_return>win_thresh).mean(),3)
        avg_5thday_posreturn = round(next_5thday_return[next_5thday_return>0].mean(),3)
        avg_5thday_negreturn = round(next_5thday_return[next_5thday_return<0].mean(),3)
        avg_5thday_expreturn = avg_5thday_posreturn*avg_5thday_winprob + avg_5thday_negreturn*(1-avg_5thday_winprob)
        
        output = pd.concat([a,b,c,next_5thday_return],axis=1)
        output.index = outlier_date
        output.columns = ['outlierday','nextday','5thday','5thday_return']

        if verbose:
            print('+1 day stock price increase probability:', round((next_day_return>win_thresh).mean(),3))
            print('+1 day stock price avg increase:', round(next_day_return[next_day_return>0].mean(),3))
            print('+1 day stock price avg decrease', round(next_day_return[next_day_return<0].mean(),3),'\n')
            print('+5 day stock price increase probability:', avg_5thday_winprob)
            print('+5 day stock price avg increase:', avg_5thday_posreturn)
            print('+5 day stock price avg decrease:', avg_5thday_negreturn)
            print('+5 day expected return:', avg_5thday_expreturn,'\n')
            return output
        
        else:
            return avg_5thday_winprob, len(output), avg_5thday_posreturn, avg_5thday_negreturn, avg_5thday_expreturn

    def record_to_df(self, record_list):
        variable = self._params['variable']
        performance = pd.DataFrame(record_list, columns=['winprob','outliernum','posret','negret','expret','/'.join(variable)])
        return performance

    def optimize_threshold(self, win_thresh):
        l = []
        for i, thresh in enumerate(itertools.product(*self._thresh_dic.values())):
            outlier = self.anomaly_dates(self._train_contract,thresh)
            stats = self.outlier_impact(self._train_contract, outlier, win_thresh=win_thresh, verbose=False)
            l.append([*stats, np.around(thresh,2)])
            if i%500 == 0:
                print(f'Testing {i}th combo')
                
        performance = self.record_to_df(l)
        return performance

    ##################################################################################
    #------------------------------- helper functions -------------------------------#
    ##################################################################################
    def pcratio(self, df):
        """
        return put-call ratio by date
        """
        ratio = df[df['type']=='put'].groupby('quotedate')['volume'].sum() / df[df['type']=='call'].groupby('quotedate')['volume'].sum()
        ratio.name = 'pcratio'
        return ratio

    def preprocess(self, df):
        """
        add pcratio column to df
        """
        pcratio = self.pcratio(df)
        return pd.merge(df, pcratio, how='left', left_on='quotedate', right_index=True)

    #contract filters
    def no_mask(self, df):
        return np.ones(len(df),dtype=bool)

    def type_contract(self, df):
        """
        return call/put filter
        """
        if self._type is None:
            return self.no_mask(df)
        else:
            return df['type']==self._type

    def expiry_contract(self, df):
        """
        return expiry in current/next/3rd month filter
        """
        if self._type is None:
            return self.no_mask(df)
        else:
            expiration, quotedate = df['expiration'], df['quotedate']
            expr_month = expiration.dt.year * 12 + expiration.dt.month
            quote_month = quotedate.dt.year * 12 + quotedate.dt.month
            if self._expiry == 'cur':
                return expr_month == quote_month
            if self._expiry == 'next':
                return expr_month - quote_month == 1
            if self._expiry == 'third':
                return expr_month - quote_month == 2

    def monthly_contract(self, df):
        """
        return monthly/non-monthly filter
        """
        if self._type is None:
            return self.no_mask(df)
        else:
            expiration= df['expiration']
            third_friday = (expiration.dt.day >=15) & (expiration.dt.day <= 21) & (expiration.dt.weekday == 4)
            if self._monthly:
                return third_friday
            else:
                return ~third_friday

    def moneyness_contract(self, df):
        """
        return atm/otm/itm filter
        """
        if self._type is None:
            return self.no_mask(df)
        else:
            delta = df['delta']
            if self._moneyness == 'atm':
                return (abs(delta) >= self._atm_delta[0]) & (abs(delta) <= self._atm_delta[1])
            if self._moneyness == 'otm':
                return abs(delta) < self._otm_delta
            if self._moneyness == 'itm':
                return abs(delta) > self._itm_delta