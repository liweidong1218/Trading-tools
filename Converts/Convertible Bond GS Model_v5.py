# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:03:44 2018

@author: Liwei Dong
"""

#%%
# Generating Coupon payment date

import numpy as np
import QuantLib as ql
from datetime import datetime
import time

# helper function

t_start = time.clock()

# Generate call/put Schedule
def dates_schedule(date1, date2):
    # discrete_schedule should be a list includes all discrete call_date
    tenor = ql.Period(ql.Daily)
    calendar = ql.UnitedStates()
    schedule = ql.Schedule(date1, date2, tenor, calendar, ql.Following,
                                   ql.Following, ql.DateGeneration.Forward, False)
    return(list(schedule))
    
def ql_to_datetime(d):
    return datetime(d.year(), d.month(), d.dayOfMonth())

def date_to_t(d):
    coupon_t = []
    for i in range(0,len(d)):
        temp = np.busday_count(ql_to_datetime(ValDate), ql_to_datetime(d[i]))/252
        coupon_t.append(temp)
        coupon_t = coupon_t[np.where(coupon_t >= 0)]
    return(coupon_t)
    
 # Function for iterative through the lattice
   
def column_calc(StockPrices_col, ConvertProb_col, y_col, ContinuationValue_col, ConversionValue_col, coupon_dates_index, convert_dates_index ,
                    call_dates_index, put_dates_index, tau, r, cs, dt,call_trigger, putPrice,callPrice):
             
    # Calculate conversion probability
    ConvertProb_col_new = [(a + b) / 2 for a, b in zip(ConvertProb_col[::1],ConvertProb_col[1::1])]
     
    # Calculate blended discount rate
    y_col_new = [(p*r + (1-p)*(r+cs)) for p in ConvertProb_col_new[::1]]
     
    # Calculate the holding value         
    ContinuationValue_col_new = [(a/(1+c*dt) + b/(1+d*dt)) / 2 for (a, b, c, d) in zip(ContinuationValue_col[::1], ContinuationValue_col[1::1], y_col[::1], y_col[1::1])] 
    
    # Coupon payment date
    if np.isin(n-1-tau, coupon_dates_index) == True:
             
             ContinuationValue_col_new = [item+Principal*(1/2*c) for item in ContinuationValue_col_new]
    
    # check put/convert schedule
    putflag = np.isin(n-1-tau, put_dates_index)
    convertflag = np.isin(n-1-tau, convert_dates_index) 
    
    for k in range(1, n+1-tau):
                  
         # check call schedule
         callflag = (np.isin(n-1-tau, call_dates_index)) & (StockPrices_col[k-1] >= call_trigger)
         
         # if t is in call date
         if (np.isin(n-1-tau, call_dates_index) == True) & (StockPrices_col[k-1] >= call_trigger):
 
             node_val = max([putPrice * putflag, ConversionValue_col[k-1] * convertflag, min(callPrice, ContinuationValue_col_new[k-1])] )
         # if t is not call date    
         else:
             
             node_val = max([putPrice * putflag, ConversionValue_col[k-1] * convertflag, ContinuationValue_col_new[k-1]] )
                    
         # 1. if Conversion happens
         if node_val == ConversionValue_col[k-1]*convertflag:
             ContinuationValue_col_new[k-1] = node_val
             ConvertProb_col_new[k-1] = 1
             
         # 2. if call or put happens
         elif node_val == putPrice*putflag or node_val == callPrice*callflag:
             ContinuationValue_col_new[k-1] = node_val
             ConvertProb_col_new[k-1] = 0
             
         else:
             ContinuationValue_col_new[k-1] = node_val       
        
    return ConvertProb_col_new, ContinuationValue_col_new, y_col_new

#%%

#issueDate = ql.Date(1, 1, 2018)
ValDate = ql.Date(31,12,2017) #
FirstCouponDate = ql.Date(7,6,2018)
maturityDate = ql.Date(7, 6, 2022)
T = np.busday_count(ql_to_datetime(ValDate), ql_to_datetime(maturityDate))/252
vol = 0.378
r = 0.0775
cs = 0.0775
S0 = 32.44
n = np.busday_count(ql_to_datetime(ValDate), ql_to_datetime(maturityDate))
if n % 2 == 0:
    n=int(n/2) # Even 
else:
    n = int(n+1/2)
#n = 400
dt = T/n
u = np.exp((r - vol**2/2)*dt + vol*np.sqrt(dt))
d = np.exp((r - vol**2/2)*dt - vol*np.sqrt(dt))
Principal = 1000
c = 0.06375  # coupon rate
ConvertRatio = 19.99632



# Coupon Date
calendar = ql.UnitedStates()
schedule = ql.Schedule(FirstCouponDate, maturityDate, ql.Period(ql.Semiannual), calendar, ql.Following,
                                   ql.Following, ql.DateGeneration.Forward, False)
coupon_t = []
for i in range(0,len(schedule)-1):
    temp = np.busday_count(ql_to_datetime(ValDate),ql_to_datetime(schedule[i]))/252
    coupon_t.append(temp)
    
coupon_dates_index = np.array([round(x/dt) for x in coupon_t])
coupon_dates_index = coupon_dates_index [coupon_dates_index >=0]
#-------------------------------------------------------------------------------------
# Conversion priods

#ConvertPrice = Principal/ConvertRatio
convert_dates_start = ql.Date(7,12,2017)
convert_dates_end = ql.Date(7,6,2022)
convert_dates_index_start = int(round(np.busday_count(ql_to_datetime(ValDate), ql_to_datetime(convert_dates_start))/252/dt))
convert_dates_index_end = int(round(np.busday_count(ql_to_datetime(ValDate), ql_to_datetime(convert_dates_end))/252/dt))
convert_dates_index = np.array(range(convert_dates_index_start, convert_dates_index_end))
convert_dates_index = convert_dates_index [convert_dates_index >=0]
 
#-------------------------------------------------------------------------------------
# Call Date and put Date
call_trigger = Principal/ConvertRatio*1.3
callPrice = 1000
call_dates_start = ql.Date(23,6,2021)
call_dates_end = ql.Date(7,6,2022)
call_dates_index_start = int(round(np.busday_count(ql_to_datetime(ValDate), ql_to_datetime(call_dates_start))/252/dt))
call_dates_index_end = int(round(np.busday_count(ql_to_datetime(ValDate), ql_to_datetime(call_dates_end))/252/dt))
call_dates_index = np.array(range(call_dates_index_start, call_dates_index_end))
call_dates_index = call_dates_index [call_dates_index >=0]

#------------------------------------------------------------------------------------
putPrice = 0 
put_dates = []#ql.Date(5,12,2019)]
put_dates_t = []
for i in range(0,len(put_dates)):
    temp = np.busday_count(ql_to_datetime(ValDate), ql_to_datetime(put_dates[i]))/252
    put_dates_t.append(temp)    
put_dates_index = np.array([round(x/dt) for x in put_dates_t])
put_dates_index = put_dates_index[put_dates_index >= 0]

#%% Stock price generation

# 1.Stock Tree
# 2.Conversion Tree
    
StockPrices = np.empty((n,n))*(np.nan)
StockPrices[n-1][0] = S0;

ConversionValue = np.empty((n,n))*(np.nan)
ConversionValue[n-1][0] = S0*ConvertRatio;


for t in range(2,n+1):
    
  for k in range(1,t+1):
      
    StockPrices[n - k][t-1] = S0*u**(k - 1) * d**(t - k)
    ConversionValue[n - k][t-1]= StockPrices[n - k][t-1] * ConvertRatio
    
    
#%% Continuation Value, Conversion probability
                
ContinuationValue = np.empty((n,n))*(np.nan)
ConvertProb = np.empty((n,n))*(np.nan)
y =  np.empty((n,n))*(np.nan)

#% Fill in the last value

for k in range(1,n+2):
    
    # If conversion happens at last node
    if ConversionValue[n-1-k][-1] >= Principal*(1+0.5*c):
        
        ContinuationValue[n-1-k][-1] = ConversionValue[n-1-k][-1];
        
        ConvertProb[n-1-k][-1] = 1
        
    # If conversion does not happen at last node    
    else:
        
        ContinuationValue[n-1-k][-1] = Principal* (1+0.5*c);
        
        ConvertProb[n-1-k][-1] = 0
        
    y[n-1-k][-1] = ConvertProb[n-1-k][-1]*r + (1- ConvertProb[n-1-k][-1]) *(r + cs) 
    
 
#%%
ConvertProb_col = ConvertProb[:, n-1]
ContinuationValue_col = ContinuationValue[:, n-1]
y_col = y[:, n-1]

for tau in range(1,n):    
    

    StockPrices_col = StockPrices[:, n-tau-1]
    ConversionValue_col = ConversionValue[:, n-tau-1]
    t_start = time.clock()
    out = column_calc(StockPrices[:, n-tau-1], ConvertProb_col, y_col, ContinuationValue_col, ConversionValue[:, n-tau-1], coupon_dates_index, convert_dates_index ,
                    call_dates_index, put_dates_index, tau, r, cs, dt,call_trigger, putPrice,callPrice)

    ConvertProb_col = out[0]
    ContinuationValue_col = out[1]
    y_col = out[2]   
    
    t_end = time.clock()
    t_end - t_start
print(ContinuationValue_col[-1])

t_end = time.clock()
t_end - t_start
 #%%       
      
