import time, os, json, logging
from helpFunc import *
import pandas as pd
import numpy as np
import datetime as dt

# print('test!')
#Create and configure logger 
log_filename = 'D:\Git Repository\_local_data_\option_data\logging\_' + dt.datetime.now().strftime('%m_%d_%Y') + ".log"
logging.basicConfig(filename=log_filename, 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
logger=logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.info("option chain saved") 

Ticker_list = ['SPY']
time_step = 5 # in minutes
month = 'Nov'

def get_chain(Ticker):
    starttime = time.ctime()
    print('Starting time:' + starttime)
    
    # Exe_path = 'CREATE DATABASE IF NOT EXISTS ' + Ticker
    for Ticker in Ticker_list:
    # Get Option Chains
        opt = get_option_chain(symbol=Ticker, includeQuotes = True)
        # time_now = dt.datetime.strptime(starttime, "%a %b %d %H:%M:%S %Y")
        
        json_path = 'D:\\Git Repository\\_local_data_\\option_data\\Json_files\\' + Ticker + '/'
        json_file = Ticker + '_' + dt.datetime.strftime(dt.datetime.strptime(starttime, "%a %b %d %H:%M:%S %Y"), "%m%d%Y %H:%M:%S").replace(':', '_') + '.txt'
        
        if not os.path.isdir(json_path):
            os.mkdir(json_path)
            with open(json_path + json_file, 'w') as write_file:
                json.dump(opt, write_file)
        else:
            with open(json_path + json_file, 'w') as write_file:
                json.dump(opt, write_file)
        
    logger.setLevel(logging.INFO)
    logger.info("option chain saved") 
    print('Ending time: ' + time.ctime())

# def test_func():
#     print('Hasta Manana!')
# while dt.datetime.strptime("06:25:00","%H:%M:%S").time() < dt.datetime.now().time() < dt.datetime.strptime("23:00:00","%H:%M:%S").time():
#     stime = time.time()
#     if dt.datetime.now().time() >= dt.datetime.strptime("06:30:00","%H:%M:%S").time():
#         try:
#             get_chain(Ticker_list)
#             # test_func()
#         except:
#             print('Error log')
#             logger.setLevel(logging.ERROR)
#             logger.error("missing data caused by error")

#             sender = 'option.chain.alert@gmail.com'
#             receiver = 'option.chain.alert@gmail.com'
#             password = 'DLWdlw110'
#             subject = 'Option chain alert'
#             message = 'Error occured at ' + dt.datetime.fromtimestamp(stime).strftime('%Y-%m-%d %H:%M:%S')

#             sendEmail(sender, password, receiver, subject, message) 
#     else:
#         print(dt.datetime.fromtimestamp(stime).strftime('%Y-%m-%d %H:%M:%S') + ' Waiting to start.')
#         logger.setLevel(logging.INFO)
#         logger.info("Waiting to start.") 
#     time.sleep(time_step*60.0 - ((time.time() - stime) % (time_step*60.0)))


while dt.datetime.strptime("06:24:59","%H:%M:%S").time() < dt.datetime.now().time() < dt.datetime.strptime("13:00:00","%H:%M:%S").time():
    stime = time.time()
    try:
        get_chain(Ticker_list)
        # test_func()
    except:
        print('Error log')
        logger.setLevel(logging.ERROR)
        logger.error("missing data caused by error")

        sender = 'option.chain.alert@gmail.com'
        receiver = 'option.chain.alert@gmail.com'
        password = 'DLWdlw110'
        subject = 'Option chain alert'
        message = 'Error occured at ' + dt.datetime.fromtimestamp(stime).strftime('%Y-%m-%d %H:%M:%S')

        sendEmail(sender, password, receiver, subject, message) 

    time.sleep(time_step*60.0 - ((time.time() - stime) % (time_step*60.0)))


sender = 'option.chain.alert@gmail.com'
receiver = 'option.chain.alert@gmail.com'
password = 'DLWdlw110'
subject = 'Option chain report'
message = 'Data streaming completed. Server closed at ' + time.ctime()

sendEmail(sender, password, receiver, subject, message) 