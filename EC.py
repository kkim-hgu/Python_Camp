#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 01:18:26 2019

@author: kkim
"""

#https://colab.research.google.com/drive/1o7Ye8Mf-dPy2gMedNUTwMHrhjvEzLhpe#scrollTo=ejfcLiCac-CV&forceEdit=true&offline=true&sandboxMode=true

import pandas as pd

ecom = pd.read_csv('Ecommerce Purchases.csv', encoding='utf-8')

#데이터프레임 확인
print(ecom.head(2))
print(ecom.info())

total = ecom.count()[0]
print("\n평균구매가격 : $",round(ecom['Purchase Price'].mean(),2))
print("최대구매가격 : $",round(ecom['Purchase Price'].max(),2))
print("최소구매가격 : $",round(ecom['Purchase Price'].min(),2))

print("\n영어사용자 수 :",ecom[ecom['Language']=='en'].count()[0])
print("변호사 수 :",ecom[ecom['Job'] == 'Lawyer'].count()[0])

#오전, 오후 구매건수 및 비율
ampm = ecom['AM or PM'].value_counts()
print("\n오전구매건수 : %d, 오전구매비율 : %.1f%%"%(ampm['AM'],100*ampm['AM']/total))
print("오후구매건수 : %d, 오후구매비율 : %.1f%%"%(ampm['PM'],100*ampm['PM']/total))

print("\n구매자 직업 랭킹 상위 1~10")
print(ecom['Job'].value_counts().head(10))
print("\n구매자 직업 랭킹 하위 1~10")
print(ecom['Job'].value_counts(sort=True, ascending=True).head(10))

# 특정 정보 찾기

# '67 EV' 지역에서 구매된 거래 정보 전체 확인하기
print("\n67 EV\n",ecom[ecom['Lot']=='67 EV'])

# '67 EV' 지역에서 구매된 거래 금액만 확인하기
print("\n67 EV\n",ecom[ecom['Lot']=='67 EV']['Purchase Price'])

#신용카드번호 4926535242672853 이용자의 이메일 주소 찾기
print("\n특정카드 이용자 이메일\n",ecom[ecom['Credit Card'] == 4926535242672853]['Email'])

#얼마나 많은 사람들이 American Express 카드로 95달러 초과로 결제하는 지 알아보기
print("\nAMEX >$95 구매자 수:",ecom[(ecom['CC Provider']=='American Express') & (ecom['Purchase Price']>95)].count()[0])

#얼마나 많은 사람들이 95달러 초과로 결제하거나 10달러 미만으로 구매하는 지 알아보기
print("\n>$95, <$10 구매자 수 :",ecom[((ecom['Purchase Price']>95) | (ecom['Purchase Price']<10))].count()[0])

print("\n사용이메일 랭킹 1~10")
print(ecom['Email'].apply(lambda x: x.split('@')[1]).value_counts().head(10))

