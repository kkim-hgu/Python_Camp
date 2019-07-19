# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
##https://medium.com/@whj2013123218/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-1%EC%9D%BC-%EC%A3%BC%EA%B0%80-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%88%98%EC%A7%91%ED%95%98%EA%B8%B0-%EB%B0%8F-%EA%B7%B8%EB%9E%98%ED%94%84-%EC%97%AC%EB%9F%AC%EA%B0%9C-%EA%B7%B8%EB%A6%AC%EA%B8%B0-4c9f803a9bbc
import pandas as pd
#import matplotlib.pyplot as plt
code='005930'
time='2019071616'
url='https://finance.naver.com/item/sise_time.nhn?code={0}&thistime={1}'.format(code, time)
df=pd.DataFrame()
for page in range(1, 41):
    pg_url = '{0}&page={1}'.format(url, page)
    df=df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)
df=df.dropna()
#
#DayIndex=pd.DataFrame()
#DayIndex['체결시각']=df.iloc[:]['체결시각']
#DayIndex['dayindex']=pd.to_datetime(DayIndex['체결시각'])
#plt.figure(figsize=(20,10))
#plt.plot(DayIndex['dayindex'],df['체결가'],c='m',label='SEC Oneday Price')
##plt.plot(DayIndex['dayIndex'],DayTelconT,c='g',label='TelCon Oneday Price')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
##print(df)
##plt.plot(df['체결시각'], df['체결가'])

