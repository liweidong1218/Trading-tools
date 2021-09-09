import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class regimeSwitch:
    def __init__(self, endog, exog=None):
        '''endog is time series, exog is predictor dataframe or series'''
        self.endog = endog
        self.exog = exog

        self.k = 0
        self.mod = 'na'
        self.modinfo = {}
        
    def __repr__(self):
        return f"Model Spec: regime:{self.k}, trend:{self.trend}"
    
    def adfTest(self):
        return adfuller(self.endog)
        
    def fit(self, k_regimes, trend='c', switch_var=True, switch_trend=True, switch_exog=True):
        '''trend: 'c' intercept,'t' time trend, 'ct' both trend, 'nc' no trend   '''
        if self.exog is not None:
            mod = sm.tsa.MarkovRegression(endog=self.endog, k_regimes=k_regimes, exog=self.exog, trend=trend, 
                                        switching_trend=switch_trend,
                                        switching_exog=switch_exog,
                                        switching_variance=switch_var).fit(search_reps=50)
        else:
            mod = sm.tsa.MarkovRegression(endog=self.endog, k_regimes=k_regimes, trend=trend, 
                                        switching_trend=switch_trend,
                                        switching_variance=switch_var).fit(search_reps=50)
        self.mod = mod
        self.modinfo['regimes'] = k_regimes
        self.modinfo['trend'] = trend
        self.modinfo['switching_trend'] = switch_trend
        self.modinfo['switching_exog'] = switch_exog
        self.modinfo['switching_variance'] = switch_var
        self.k = k_regimes
        self.trend = trend
        
    def plotSeries(self, size=(10,6)):
        '''won't work if average price dataset has time object as index'''
        fig, axe = plt.subplots(figsize=size)
        axe.plot(self.endog)
        axe.set(title='Time Series Plot')

    def lastProb(self):
        '''probability in each regime for last time step in training set'''
        lastProb = []
        for i in range(self.k):
            lastProb.append(self.mod.smoothed_marginal_probabilities[i][-1])
        return np.array(lastProb)
        
    def tranMatrix(self):
        '''k*k transition matrix, Pij is probability i->j'''
        params = self.mod.params
        tparam = np.array(params[params.index.str.contains('p')]).reshape(self.k-1, self.k)
        tranM = np.empty((self.k, self.k))
        tranM[:self.k-1] = tparam
        tranM[-1] = 1 - tparam.sum(axis=0)
        
        return tranM.T
        
    def betaMatrix(self):
        '''k*k beta matrix, each row contains one beta under each regime'''
        if self.exog is not None:
            params = self.mod.params
            if self.trend == 'nc':
                #if no trend, parameter name is same as exog variable name
                #else, parameter name is changed to x by default
                exog_names = self.exog.columns
                xparamIndex = params.index.str.contains('|'.join(exog_names))
            else:
                xparamIndex = params.index.str.contains('x')
            xparam = params[xparamIndex]
            nExog = int(len(xparam) / self.k)
            xparam = np.array(xparam).reshape(nExog, self.k)
            return xparam
        else:  
            return 0
        
    def interceptMatrix(self):
        params = self.mod.params
        cparam = params[params.index.str.contains('c')]
        cparam = np.array(cparam)
        if cparam.size == 0:
            #if no constant parameter, set it to 0
            cparam = np.array([0])

        return cparam
            
    def forecast(self, periods=None, exogTest=None):
        '''
        either input # of periods to forecast if there is no exogenous test set,
        or input exogenous test set.

        return predicted value, predicted probability for subsequent periods
        '''
        tranMatrix = self.tranMatrix()
        betaMatrix = self.betaMatrix()
        constMatrix = self.interceptMatrix()
        if exogTest is None:
            exogTest = pd.Series([0]*periods)
        if periods is None:
            periods = len(exogTest)
            
        forecastProb = [self.lastProb()]
        predictValue = []
        
        for p in range(periods):
            newProb = np.dot(forecastProb[p],tranMatrix)
            yUnweighted = np.dot(exogTest.iloc[p], betaMatrix) + constMatrix
            yWeighted = np.dot(yUnweighted, newProb)
            predictValue.append(yWeighted)
            forecastProb.append(newProb)
            
        if exogTest is not None:
            predictValue = pd.Series(predictValue, index=exogTest.index)
            forecastProb = pd.DataFrame(forecastProb[1:], index=exogTest.index)
        
        return predictValue, forecastProb
        
    @classmethod
    def MSE(cls, actual, predict):
        return np.mean(np.power(actual - predict,2))
    
    @classmethod
    def plotProb(cls, series, probDF):
        obs, k = probDF.shape
        fig, axes = plt.subplots(k, figsize=(12,4*k))
        for i in range(k):
            ax1 = axes[i] 
            ax1.plot(probDF[i],label='Probability') 
            ax2 = ax1.twinx()
            ax2.plot(series,'r--',label='Time Series')
            ax1.set(title=f'Smoothed probability of variance {i} regime')
            if i==0:
                fig.legend(loc=1)
                    
    @classmethod        
    def plotPredict(cls, actual, predict, title):
        plt.figure(figsize=(12,6))
        plt.plot(predict,'r--')
        plt.plot(actual)
        plt.legend(['predict','actual'])
        plt.title(title)
        
    @classmethod        
    def plotCompareModel(cls, actual, predict, title):
        plt.figure(figsize=(12,6))
        plt.plot(predict[0],'r--')
        plt.plot(predict[1],'m')
        plt.plot(actual)
        plt.legend(['M0','M1','actual'])
        plt.title(title)     


#use df downloaded from yahoofinance to initialize
#create variables for MRS model
class mrsPreprocess:
    def __init__(self, df, lag):
        self.df = self.create_lag_variable(df, lag)
        self.train = None
        self.test = None

    #construct variables
    def create_lag_variable(self, df, lag=2):
        """
        create endog and exog variables for markov regime switching model
        endog: log return
        exog: 2 lagged return, volume gap, fraction high, fraction low
        """
        self.assert_columns(df)
        temp_df = df.copy()
        #endogenous
        temp_df['pricechange'] = temp_df['close'].diff()
        temp_df['ret'] = np.log(temp_df['close']).diff()
        #exogenous
        for l in range(lag):
            name = 'lag_{}_ret'.format(l+1)
            temp_df[name] = temp_df['ret'].shift(l+1)
        temp_df.dropna(inplace=True)
        return temp_df
        
    def group_data():
        pass

    def assert_columns(self, df):
        """
        check if df has desired columns
        """
        columns = {'open','high','low','close','volume', 'trade_count'}
        missing_columns = columns.symmetric_difference(set(df.columns))
        assert columns.issubset(df.columns), print(missing_columns,' missing')

    def split_train_test(self, split_pct = 0.8, col = ['ret']):
        split_num = round(self.df.shape[0] * split_pct)
        self.train = self.df[col].iloc[:split_num]
        self.test = self.df[col].iloc[split_num:]
        return self.train, self.test

"""
get 3 regime switching models' stats and predicted values for 1 stock
model0: endog
model1: endog + lagged return
"""

def get_MRS_stats(train_endog, train_exog, test_endog, test_exog):

    actual = test_endog
    rmse = []
    model = []
    predictV = []
    predictP = []

    for i in range(2): # test 2 models
        if i==0:   
            mrs = regimeSwitch(train_endog)
        else:      
            mrs = regimeSwitch(train_endog, train_exog)
        
        #start with 3 regimes, if not converge, try 2 regimes
        try:
            mrs.fit(k_regimes=3, trend='c',switch_var=True, switch_trend=True, switch_exog=True)
        except:
            mrs.fit(k_regimes=2, trend='c',switch_var=True, switch_trend=True, switch_exog=True)
        
        if i==0:   
            predictValue, predictProb = mrs.forecast(periods=len(test_endog))
            predictValue.index = test_endog.index
            predictProb.index = test_endog.index
        else: 
            predictValue, predictProb = mrs.forecast(exogTest = test_exog)

        predictV.append(predictValue)
        predictP.append(predictProb)
        rmse.append(regimeSwitch.MSE(test_endog, predictValue)**0.5)
        model.append(mrs)
        
    return predictV, predictP, rmse, actual, model