import time, os, json
from helpFunc import *
import pandas as pd
import numpy as np
from td.client import TDClient
from datetime import date
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('ggplot')

def get_ratio(Ticker, dates):    
#     # Create new session
#     TDSession = TDClient(
#     client_id= 'UPGQCPR4GR9EGFDMUADAM0N8ICKGVMXO',
#     redirect_uri = 'http://localhost',
#     credentials_path = 'C:/Users/liwei/Desktop/ToDo/Projects/Trading System/System Tools/API/td_state.json')
#     TDSession.login()
    
#     # Option Chain Example
#     opt_chain = {
#         'symbol': Ticker,
#         'includeQuotes': True
#     }

    # Get Option Chains
    opt = get_option_chain(symbol=Ticker, includeQuotes = True)
    # Get put/call ratios at certain expiry
    exp_date = getExpiry(opt) # optimize this function
    time_now = dt.datetime.now().date()
    exp_list = [dt.datetime.strptime(date, "%Y-%m-%d").date() for date in [x.split(':')[0] for x in exp_date]]
    date_indx = [x for x, val in enumerate(exp_list) if val == dt.datetime.strptime(dates, '%Y-%m-%d').date()][0]
    ratio = getRatio(opt, exp_date[date_indx])
    print(exp_date[date_indx])
    return ratio


def animate(i, timestamp, ratio, ticker_seq, dates_seq, keys, title, ylabel):

    num_of_plot = len(keys)

    for i,v in enumerate(range(num_of_plot)): 

        timestamp_temp = dt.datetime.now().strftime('%H:%M:%S')
        timestamp[keys[i]].append(timestamp_temp)
        ratio_temp = get_ratio(ticker_seq[i], dates_seq[i])
        ratio[keys[i]].append(ratio_temp)
        v += 1
        plt.subplot(num_of_plot, 1, v)
        plt.plot(timestamp[keys[i]], ratio[keys[i]], 'bo-')

        plt.xticks(rotation=45, ha='right')
        plt.title(title[i])
        plt.ylabel(ylabel[i])

    print([len(m) for m in ratio], [len(n) for n in timestamp])

    #plt.subplot(212)
    # plt.plot(timestamp, ratio, 'ro-')
    # # Format plot
    # plt.xticks(rotation=45, ha='right')
    # # #plt.subplots_adjust(bottom=0.30)
    # plt.title(title)
    # plt.ylabel(ylabel)

    # import matplotlib.pyplot as plt
    # from pylab import *
    # import numpy as np

    # x = np.linspace(0, 2*np.pi, 400)
    # y = np.sin(x**2)

    # subplots_adjust(hspace=0.000)
    # number_of_subplots=3

    # for i,v in enumerate(xrange(number_of_subplots)):
    #     v = v+1
    #     ax1 = subplot(number_of_subplots,1,v)
    #     ax1.plot(x,y)

    # plt.show()


ticker_list = ['SPY', 'AMZN']
dates_list = ['2021-01-15']
fig = plt.figure(figsize=(10, 10))

ticker_seq = sorted(ticker_list * len(dates_list))
dates_seq = dates_list * len(ticker_list)
keys = [x + '_' + str(y) for (x, y) in zip(ticker_seq, dates_seq)]
ratio = {key: [] for key in keys}
timestamp = {key: [] for key in keys}

title = ["Put/Call Ratio (Volume) " + date for date in dates_seq]
ylabel = [ticker + " Ratio" for ticker in ticker_seq]
ani = animation.FuncAnimation(fig, animate, fargs=(timestamp, ratio, ticker_seq, dates_seq, keys, title, ylabel), interval=10000) # interval in mili seconds
plt.show()
