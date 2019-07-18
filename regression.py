#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 01:51:13 2019

@author: kkim
"""

#Regression
#source from https://github.com/PinkWink/DataScience/blob/master/source_code/07.%20Time%20Series%20Data%20Handle.ipynb

#import pandas as pd
#import pandas_datareader.data as web
#import numpy as np
#import matplotlib.pyplot as plt
#
#from prophet import Prophet
#from datetime import datetime
#
#pinkwink_web = pd.read_csv('https://raw.githubusercontent.com/PinkWink/DataScience/master/data/08.%20PinkWink%20Web%20Traffic.csv', encoding='utf-8', thousands=',', names = ['date','hit'], index_col=0)
#pinkwink_web = pinkwink_web[pinkwink_web['hit'].notnull()]
#print(pinkwink_web.head())
#
#pinkwink_web['hit'].plot(figsize=(12,4), grid=True);
##
##구간 설정
#time = np.arange(0,len(pinkwink_web))
#traffic = pinkwink_web['hit'].values
#
#fx = np.linspace(0, time[-1], 1000)

# 1,2,3,15차 함수로 회귀
#fp1 = np.polyfit(time, traffic, 1)
#f1 = np.poly1d(fp1)
#
#f2p = np.polyfit(time, traffic, 2)
#f2 = np.poly1d(f2p)
#
#f3p = np.polyfit(time, traffic, 3)
#f3 = np.poly1d(f3p)
#
#f15p = np.polyfit(time, traffic, 15)
#f15 = np.poly1d(f15p)
#
#plt.figure(figsize=(12,4))
#plt.scatter(time, traffic, s=10)
#
#plt.plot(fx, f1(fx), lw=4, label='f1')
#plt.plot(fx, f2(fx), lw=4, label='f2')
#plt.plot(fx, f3(fx), lw=4, label='f3')
#plt.plot(fx, f15(fx), lw=4, label='f15')
#
#plt.grid(True, linestyle='-', color='0.75')
#
#plt.legend(loc=2)
#plt.show()

##
## 주식 데이터 가져오기 
#from pandas_datareader import data
#import matplotlib.pyplot as plt
#import fix_yahoo_finance as yf
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()
#
#yf.pdr_override()
#
#start_date = '2000-1-1' 
#end_date = '2019-6-30' 
#
#NAVER = data.get_data_yahoo('035420.KS', start_date, end_date)
#HCON = data.get_data_yahoo('000720.KS', start_date, end_date)
#
#print(NAVER.columns)
#
#plt.figure(figsize=(12,6))
#plt.plot(NAVER['Close'], label='Naver')
#plt.plot(HCON['Close'], label='Hyundai Con.')

#plt.legend(loc=2)
#
#plt.grid()
#plt.show()
#
#
#
#
