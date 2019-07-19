#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 00:04:31 2019

@author: kkim
"""
##https://kind.krx.co.kr/corpgeneral/corpList.do?method=loadInitPage#
##http://www.check.co.kr/download?type=manual&fileName=JongMokCodeTable_20140616_Uodate02.xlsx
#import pandas as pd
#url = 'http://www.check.co.kr/download?type=manual&fileName=JongMokCodeTable_20140616_Uodate02.xlsx'
#df = pd.read_excel(url, header=3)
#print(df.head())
#print(df.size)

#import pandas as pd
#url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
#df = pd.read_html(url, header=0)[0]
#df['종목코드'] = df['종목코드'].map('{:06d}'.format)
#print(df.head())


#https://minjejeon.github.io/learningstock/2017/09/07/download-krx-ticker-symbols-at-once.html
import urllib.parse
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt

MARKET_CODE_DICT = {
    'kospi': 'stockMkt',
    'kosdaq': 'kosdaqMkt',
    'konex': 'konexMkt'
}

DOWNLOAD_URL = 'kind.krx.co.kr/corpgeneral/corpList.do'

def download_stock_codes(market=None, delisted=False):
    params = {'method': 'download'}

    if market.lower() in MARKET_CODE_DICT:
        params['marketType'] = MARKET_CODE_DICT[market]

    if not delisted:
        params['searchType'] = 13

    params_string = urllib.parse.urlencode(params)
    request_url = urllib.parse.urlunsplit(['http', DOWNLOAD_URL, '', params_string, ''])

    df = pd.read_html(request_url, header=0)[0]
    df.종목코드 = df.종목코드.map('{:06d}'.format)

    return df

kosdaq_stocks = download_stock_codes('kosdaq')
print(kosdaq_stocks.head())
print(kosdaq_stocks.size)

kospi_stocks = download_stock_codes('kospi')
print(kospi_stocks.head())
print(kospi_stocks.size)

results = {}
for code in kosdaq_stocks['종목코드'].head():
    results[code] = data.DataReader(code + '.KQ', 'yahoo', '2017-01-01', None)

df = pd.concat(results, axis=1)
print(df.loc[:, pd.IndexSlice[:, 'Adj Close']].tail())

results2 = {}
for code in kospi_stocks['종목코드'].head():
    results2[code] = data.DataReader(code + '.KS', 'yahoo', '2017-01-01', None)

df2 = pd.concat(results2, axis=1)
print(df2.loc[:, pd.IndexSlice[:, 'Adj Close']].tail())
plt.figure(figsize=(12,8))
plt.plot(df2.loc[:, pd.IndexSlice[:, 'Adj Close']])

#### conda install -c anaconda pandas-datareader
#import pandas_datareader.data as web
##import numpy as np
#import matplotlib.pyplot as plt
#
#start = '2015-01-01'
#end = '2019-07-18'
#code= '207940.KS'
#company = 'Samsung BIO'
#df = web.DataReader(code,'yahoo',start,end)
#print(df.head())
#plt.plot(df['Close'], label=company)
#plt.legend()

 