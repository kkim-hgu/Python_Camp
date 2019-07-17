#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 00:46:26 2019

@author: kkim
"""
## Simple Text File Processing
## 1. Make a random score file with 3 numbers
#from random import randint, choice
#import string
#
#f = open("score1.txt", "w")
#for i in range(100):
#    f.write("%d %d %d\n" % (randint(50, 100), randint(50, 100), randint(50, 100)))
#f.close()
#
#
## 2. Make a random score file with name & 3 numbers
#from random import randint, choice
#import string
#
#f = open("score2.txt", "w")
#for i in range(100):
#    randomstr = ''.join(choice(string.ascii_uppercase) for _ in range(3))
#    f.write("%s %d %d %d\n" % (randomstr, randint(50, 100), randint(50, 100), randint(50, 100)))
#f.close()
#
## 3. Make a random score file with name & 3 numbers (comma separated)
#from random import randint, choice
#import string
#
#f = open("score3.txt", "w")
#for i in range(100):
#    randomstr = ''.join(choice(string.ascii_uppercase) for _ in range(3))
#    f.write("%s,%d,%d,%d\n" % (randomstr, randint(50, 100), randint(50, 100), randint(50, 100)))
#f.close()

## 4. Make a random score file with headers, name & 3 numbers
#from random import randint, choice
#import string
#
#f = open("score4.txt", "w")
#f.write('score file\n')
#f.write('Name Kor Eng Mat\n')
#for i in range(100):
#    randomstr = ''.join(choice(string.ascii_uppercase) for _ in range(3))
#    f.write("%s %d %d %d\n" % (randomstr, randint(50, 100), randint(50, 100), randint(50, 100)))
#f.close()


## 4. Read a score file.
## 1번 방법 : 숫자만 있는 경우
#import numpy as np
#
#scores = np.genfromtxt('score1.txt', names=('KOR','ENG','MAT'))
#print(scores)
#print(scores['ENG'])

## 2번 방법 : 문자열과 숫자가 섞여 있는 경우
#import numpy as np
#
#scores = np.genfromtxt('score2.txt', names=('Name','KOR','ENG','MAT'), dtype=None, encoding=None)
#print(scores)
#print(scores['Name'])
#print(scores.dtype)

## 3번 방법 : 문자열과 숫자가 컴마로 구분된 경우 
#import numpy as np
#
#scores = np.genfromtxt('score3.txt', names=('Name','KOR','ENG','MAT'), delimiter=',', dtype=None, encoding=None)
#print(scores)
#print(scores['Name'])
#print(scores.dtype)

## 4번 방법 : Header 있는 경우
#import numpy as np
#scores = np.genfromtxt('score4.txt', names=('Name','KOR','ENG','MAT'), dtype=None, skip_header=2, encoding=None)
#print(scores)
#print(scores.size)
#print(scores['Name'])
#print(scores.dtype)

## 5번 방법 : 문자열과 숫자가 컴마로 구분된 경우  
#import pandas as pd
#pdscores = pd.read_csv('score3.txt',sep=",",header=None)
#pdscores.columns=['Name','KOR','ENG','MAT']
#print(pdscores)


########## pandas

#◉ Series
# 

#import pandas as pd 
#data = [77,66,88,99,55] # 그냥 리스트 
#obj = pd.Series(data) 
#print(obj) 
#print(obj.values) 
#print(obj.index)  # range(0,5,1) 
#print(obj[2])
#
#• 사용자 정의 index를 사용할 수 있음
#import pandas as pd
#data = [77,66,88,99,55]
#keys = [5,'Lee','Park','Jang','Hwang'] 
#obj = pd.Series(data, index=keys) 
#print(obj)
#print(obj.values)
#print(obj.index)
#print(obj[5])
#obj['Kim'] = 95
#print(obj)
#print(obj['Kim']) 
#print(obj[['Kim','Lee','Hwang']]) 
#print(obj[obj > 80])
#print('Oh' in obj)
#
#• 딕셔너리를 통해 Series 만들기
#import pandas as pd
#data = {'Kim':77,'Lee':66,'Park':88,'Jang':99,'Hwang':55} 
#obj = pd.Series(data)
#print(obj)
#
#• Series 간 연산
#import pandas as pd
#data1 = {'Kim':77,'Lee':66,'Park':88,'Jang':99,'Hwang':55} 
#obj1 = pd.Series(data1)
#data2 = {'Kim':87,'Lee':87,'Park':80,'Jang':69,'Hwang':85} 
#obj2 = pd.Series(data2)
#print(obj1 + obj2)
#
#• reindex
#import pandas as pd
#data1 = {'Kim':77,'Lee':66,'Park':88,'Jang':99,'Hwang':55} 
#obj1 = pd.Series(data1)
#print(obj1)
#obj2 = obj1.reindex(['Kim','Park','Lee','Jang','Hwang']) 
#print(obj2)
## 
#◉ DataFrame
#index를 공통으로 사용하고 있는 Series 객체들을 담고 있다고 생각하면 된다.
#
#• 딕셔너리로 Column 정보를 넣어서 생성
#import pandas as pd
##import matplotlib.pyplot as plt
#data = {'col1': [77,66,88,99,55],'col2': [45,56,67,78,89], 'col3': [97,86,75,64,53]}
#df = pd.DataFrame(data) 
#print(df)
#df.plot()
#
##• index 명 삽입
#import pandas as pd
##import matplotlib.pyplot as plt
#data = {'col1': [77,66,88,99,55],'col2': [45,56,67,78,89],'col3': [97,86,75,64,53]}
#keys = ['Kim','Lee','Park','Jang','Hwang']
#df = pd.DataFrame(data, index=keys) 
#print(df)
#df.plot()
#
#• Column 별로 만든 리스트를 묶어서 생성
#import pandas as pd
#import matplotlib.pyplot as plt
#kor = [77,66,88,99,55]
#eng = [45,56,67,78,89]
#mat = [97,86,75,64,53]
#data = list(zip(kor,eng,mat))
#keys = ['Kim','Lee','Park','Jang','Hwang']
#df = pd.DataFrame(data, index=keys, columns=['KOR','ENG','MAT']) 
#print(df)
##df.plot()
#plt.plot(df)
#
#• 같은 키를 사용하는 여러 딕셔너리를 조합하여 생성
#import pandas as pd
#kor = {'Kim':77,'Lee':66,'Park':88,'Jang':99,'Hwang':55} 
#eng = {'Kim':45,'Lee':56,'Park':67,'Jang':78,'Hwang':89} 
#mat = {'Kim':97,'Lee':86,'Park':75,'Jang':64,'Hwang':53} 
#data = {'KOR':kor, 'ENG':eng, 'MAT':mat}
#df = pd.DataFrame(data)
#print(df)
#
#• 기존 Column을 이용하여 새로운 Column 추가
#import pandas as pd
#kor = {'Kim':77,'Lee':66,'Park':88,'Jang':99,'Hwang':55} 
#eng = {'Kim':45,'Lee':56,'Park':67,'Jang':78,'Hwang':89} 
#mat = {'Kim':97,'Lee':86,'Park':75,'Jang':64,'Hwang':53} 
#data = {'KOR':kor, 'ENG':eng, 'MAT':mat}
#df = pd.DataFrame(data)
#print(df)
#df['Sum']=df['KOR']+df['ENG']+df['MAT'] 
#df['Avg']=df['Sum']/3
#print(df)
#
##• reindex
#import pandas as pd
#kor = {'Kim':77,'Lee':66,'Park':88,'Jang':99,'Hwang':55}
#eng = {'Kim':45,'Lee':56,'Park':67,'Jang':78,'Hwang':89}
#mat = {'Kim':97,'Lee':86,'Park':75,'Jang':64,'Hwang':53}
#data = {'KOR':kor, 'ENG':eng, 'MAT':mat}
#df = pd.DataFrame(data)
#df['Sum']=df['KOR']+df['ENG']+df['MAT']
#df['Avg']=df['Sum']/3
#print(df)
#df2 = df.reindex(index=['Kim','Lee','Park','Jang','Hwang'],columns=['KOR','ENG','MAT','Sum','Avg'])
#print(df2)
#
#• 정렬
#import pandas as pd
#kor = [77,66,88,99,55]
#eng = [45,56,67,78,89]
#mat = [97,86,75,64,53]
#data = list(zip(kor,eng,mat))
#keys = ['Kim','Lee','Park','Jang','Hwang']
#df = pd.DataFrame(data, index=keys, columns=['KOR','ENG','MAT']) 
#df['Sum']=df['KOR']+df['ENG']+df['MAT']
#df['Avg']=df['Sum']/3
#print(df)
#df2 = df.sort_values('Sum', ascending=False) 
#print(df2)
#

### Reading from internet file
#import pandas as pd
#import matplotlib.pyplot as plt
#df = pd.read_csv('https://raw.githubusercontent.com/datascienceschool/docker_rpython/master/data/titanic.csv')
##print(df)
#print(df.head(10))
#print(df.tail(3))
#plt.plot(df['Survived'])


##두 DataFrame 병합
##source from https://github.com/PinkWink/DataScience/blob/master/source_code/01.%20Basic%20of%20Python%2C%20Pandas%20and%20Matplotlib%20%20via%20analysis%20of%20CCTV%20in%20Seoul.ipynb
##
#import pandas as pd
##행병합
#df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 
#                    'B': ['B0', 'B1', 'B2', 'B3'],
#                    'C': ['C0', 'C1', 'C2', 'C3'],
#                    'D': ['D0', 'D1', 'D2', 'D3']},
#                   index=[0, 1, 2, 3])
#
#df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
#                    'B': ['B4', 'B5', 'B6', 'B7'],
#                    'C': ['C4', 'C5', 'C6', 'C7'],
#                    'D': ['D4', 'D5', 'D6', 'D7']},
#                   index=[4, 5, 6, 7])
#
#df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
#                    'B': ['B8', 'B9', 'B10', 'B11'],
#                    'C': ['C8', 'C9', 'C10', 'C11'],
#                    'D': ['D8', 'D9', 'D10', 'D11']},
#                   index=[8, 9, 10, 11])

#print(df1)
#print(df2)
#print(df3)
#
#M1 = pd.concat([df1, df2, df3])
#print(M1)
#
#M1 = pd.concat([df1, df2, df3], keys=['df1', 'df2', 'df3'])
#print(M1)
##
#print(M1.index)
#print(M1.index.get_level_values(1))
#
### 열병합
#
#df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'], 
#                    'D': ['D2', 'D3', 'D6', 'D7'],
#                    'F': ['F2', 'F3', 'F6', 'F7']},
#                   index=[2, 3, 6, 7])
#print(df1)
#print(df4)
#M2 = pd.concat([df1, df4], axis=1)
#print(M2)
#
#M2 = pd.concat([df1, df4], axis=1, join='inner')
#print(M2)
#
#M2 = pd.concat([df1, df4], axis=1, join_axes=[df4.index])
#print(M2)
#
#M2 = pd.concat([df1, df4], ignore_index=True)
#print(M2)
#
##Merge
#
#left = pd.DataFrame({'key': ['K0', 'K4', 'K2', 'K3'],
#                     'A': ['A0', 'A1', 'A2', 'A3'],
#                     'B': ['B0', 'B1', 'B2', 'B3']})
#
#right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
#                      'C': ['C0', 'C1', 'C2', 'C3'],
#                      'D': ['D0', 'D1', 'D2', 'D3']})
#
#print(left)
#print(right)
#
#M3 = pd.merge(left, right, on='key')
#print(M3)
#
#M3 = pd.merge(left, right, how='left', on='key')
#print(M3)
#
#M3 = pd.merge(left, right, how='right', on='key')
#print(M3)
#
#M3 = pd.merge(left, right, how='inner', on='key')
#print(M3)
#
#M3 = pd.merge(left, right, how='outer', on='key')
#print(M3)



########## 원본파일 - 포항시 주차장 정보
### https:// www.data.go.kr/dataset/15009552/fileData.do

##1 simple_parsing_and_write
#input_file = "p_20180930.csv" #포항시 주차장 정보 파일 
#output_file = "p_out1.txt" # 출력파일#1
#
#filereader = open(input_file, 'r', newline='')
#filewriter = open(output_file, 'w', newline='')
#header = filereader.readline() #first line
#header = header.strip()
#header_list = header.split(',')
#print(header_list)
#filewriter.write('/'.join(header_list)+'\n')
#for row in filereader:  # repeat reading a row
#	row = row.strip()
#	row_list = row.split(',')
#	#print(row_list)
#	print('/'.join(row_list))
#	filewriter.write('/'.join(row_list)+'\n')
#filereader.close()
#filewriter.close()


## #2 CSV module
#import csv
#
#input_file = "p_20180930.csv"
#output_file = "p_out2.txt"
#
#csv_in_file = open(input_file, 'r', newline='')
#csv_out_file = open(output_file, 'w', newline='')
#filereader = csv.reader(, delimiter=',')
#filewriter = csv.writer(csv_out_file, delimiter='/')
#for row_list in filereader:
#	filewriter.writerow(row_list)
#csv_in_file.close()
#csv_out_file.close()
#			

## #3 pandas r/w
#import pandas as pd
#
#input_file = "p_20180930.csv"
#output_file = "p_out3.txt"
#
#data_frame = pd.read_csv(input_file)
#print(data_frame)
#data_frame.to_csv(output_file, index=False)


## #4 Value in Row Meets a Condition			
#
#import csv
#
#input_file = "p_20180930.csv"
#output_file = "p_out4.txt"
#
#csv_in_file = open(input_file, 'r', newline='')
#csv_out_file = open(output_file, 'w', newline='')
#filereader = csv.reader(csv_in_file)
#filewriter = csv.writer(csv_out_file)
#header = next(filereader)
#filewriter.writerow(header)
#for row_list in filereader:
#	ptype = str(row_list[3]).strip()
#	psize = str(row_list[6]).replace(',', '')
#	if ptype == '노상' and int(psize) > 30:
#		filewriter.writerow(row_list)
#csv_in_file.close()
#csv_out_file.close()

## #5 pandas Value in Row Meets a Condition			
#import pandas as pd
#
#input_file = "p_20180930.csv"
#output_file = "p_out5.txt"
#
#data_frame = pd.read_csv(input_file)
#
#data_frame['주차구획수'] = data_frame['주차구획수'].astype(float)
#my_condition = data_frame.loc[(data_frame['주차장유형'].str.contains('노상')) & (data_frame['주차구획수'] > 30), :]
#
#my_condition.to_csv(output_file, index=False)


## #6 Value in Row Matches a Pattern
#import pandas as pd
#
#input_file = "p_20180930.csv"
#output_file = "p_out6.txt"
#
#data_frame = pd.read_csv(input_file)
#my_pattern = data_frame.loc[data_frame['소재지도로명주소'].str.startswith("경상북도 포항시 북구"), :]
#
#my_pattern.to_csv(output_file, index=False)


########## - 포항시 인구정보 통계 
### https://www.data.go.kr/dataset/15030316/fileData.do

## #7 Value in Row Is in a Set of Interest
#import csv
#important_dates = ['2010.01', '2011.01']
#input_file = "p_20180630.csv"
#output_file = "p_out7.txt"
#
#csv_in_file = open(input_file, 'r', newline='')
#csv_out_file = open(output_file, 'w', newline='')
#filereader = csv.reader(csv_in_file)
#filewriter = csv.writer(csv_out_file)
#header = next(filereader)
#filewriter.writerow(header)
#for row_list in filereader:
#	a_date = row_list[0]
#	if a_date in important_dates:
#		filewriter.writerow(row_list)
#csv_in_file.close()
#csv_out_file.close()

## #8 Value in Row Is in a Set of Interest
#import pandas as pd
#
#important_dates = ['2010.01', '2011.01']			
#input_file = "p_20180630.csv"
#output_file = "p_out8.txt"
#
#data_frame = pd.read_csv(input_file)
#my_set = data_frame.loc[data_frame['년월'].isin(important_dates), :]
#my_set.to_csv(output_file, index=False)


## #9 Using column index values
#import csv
#my_columns = [0,1,2]
#input_file = "p_20180630.csv"
#output_file = "p_out9.txt"
#
#csv_in_file = open(input_file, 'r', newline='')
#csv_out_file = open(output_file, 'w', newline='')
#filereader = csv.reader(csv_in_file)
#filewriter = csv.writer(csv_out_file)
#for row_list in filereader:
#	row_list_output = [ ]
#	for index_value in my_columns:
#		row_list_output.append(row_list[index_value])
#	filewriter.writerow(row_list_output)
#csv_in_file.close()
#csv_out_file.close()


## #10 pandas Using column index values
#import pandas as pd
#
#input_file = "p_20180630.csv"
#output_file = "p_out10.txt"
#my_columns = [0,1,2]
#data_frame = pd.read_csv(input_file)
#my_index = data_frame.iloc[:, my_columns]
#
#my_index.to_csv(output_file, index=False)




## #11 Using column headings
#import csv
#my_columns = ['년월', ' 한국인남 ' , ' 한국인여 ']
#my_columns_index = []
#
#input_file = "p_20180630.csv"
#output_file = "p_out11.txt"
#
#csv_in_file = open(input_file, 'r', newline='')
#csv_out_file = open(output_file, 'w', newline='')
#filereader = csv.reader(csv_in_file)
#filewriter = csv.writer(csv_out_file)
#header = next(filereader)
#for index_value in range(len(header)): # finding column index
#	if header[index_value] in my_columns:
#		my_columns_index.append(index_value)
#filewriter.writerow(my_columns)
#for row_list in filereader:
#	row_list_output = [ ]
#	for index_value in my_columns_index:
#		row_list_output.append(row_list[index_value])
#	filewriter.writerow(row_list_output)
#csv_in_file.close()
#csv_out_file.close()


## #12 pandas Using column headings
#import pandas as pd
#
#input_file = "p_20180630.csv"
#output_file = "p_out12.txt"
#my_columns = ['년월', ' 한국인남 ' , ' 한국인여 ']
#
#data_frame = pd.read_csv(input_file)
#my_data = data_frame.loc[:, my_columns]
#
#my_data.to_csv(output_file, index=False)





###########################################
#Matplotlib
#
#◉ plot()
#import matplotlib.pyplot as plt 
#import numpy as np
##data=[1,2,3,4,5,6,7] 
##data=[4,5,6,1,2,3,7,6,1,2,3] 
#data = np.arange(1,20,0.1)
#plt.plot(data) 
#plt.ylabel('Values')
#plt.show()

#import matplotlib.pyplot as plt 
#import numpy as np
#x = np.arange(0,10,0.1)
#y = np.sin(x)
#plt.figure(figsize=(10,4))
#plt.plot(x,y) 
#plt.grid()
#plt.xlabel('x')
#plt.ylabel('sin')
#plt.show()

# 
#import numpy as np
#import matplotlib.pyplot as plt 
#import seaborn as sns 
#sns.set()
#x_data = np.linspace(0,10,500) 
#y_data = np.power(x_data,2) 
#plt.figure(figsize=(10,10))
#plt.plot(x_data, y_data) 
#plt.ylabel('Values')
#plt.show()
#
#• 특정 좌표를 이어주는 그래프 그리기
#import matplotlib.pyplot as plt 
#import seaborn as sns 
#sns.set()
#x_data = [0,1,2,3,4,5,6] 
#y_data = [1,3,4,7,10,15,16] 
#colormap = x_data
#plt.plot(x_data, y_data) 
##plt.scatter(x_data, y_data, s=7, c=colormap, marker='>')
##plt.bar(x_data, y_data, width=0.2, color='m')
#plt.ylabel('Values')
#plt.show()
#
#• 그래프 2개 그리기
#import numpy as np
#import matplotlib.pyplot as plt 
#import seaborn as sns 
#sns.set()
#x_data = np.array([0,1,2,3,4,5,6]) 
#y_data1 = np.array([1,3,4,7,10,15,16]) 
#y_data2 = np.array([5,4,2,9,15,25,36])
#plt.figure(figsize=(10,10))
##plt.plot(x_data, y_data1) 
##plt.plot(x_data, y_data2) 
##plt.bar(x_data, y_data1, width=0.3, color='r', label="Graph #1") 
##plt.bar(x_data+0.5, y_data2, width=0.3, color='g', label="Graph #2") 
##plt.barh(x_data, y_data1, color='r', label="Graph #1") 
##plt.barh(x_data, -y_data2, color='g', label="Graph #2") 
#plt.ylabel('Values')
#plt.pie(y_data1)
#plt.legend()
#plt.show()
#
#• 좌표 찍기
#import numpy as np
#import matplotlib.pyplot as plt 
#import seaborn as sns
#sns.set()
#x_values = np.linspace(0, 10, 20) 
#y_values = np.zeros(20) 
#print(x_values)
#print(y_values)
#plt.figure(figsize=(10,10))
#plt.plot(x_values, y_values, 'o') 
#plt.plot(x_values, y_values+0.5, 'r^') 
#plt.ylim([-1,1])
#plt.show()

#legend()
#import numpy as np
#import matplotlib.pyplot as plt 
#import seaborn as sns
#sns.set()
#x_values = np.linspace(0, 10, 50) 
#y_values1 = np.sin(x_values) 
#y_values2 = np.cos(x_values) 
#plt.figure(figsize=(10,10))
#plt.plot(x_values, y_values1, 'o', label='sin') 
#plt.plot(x_values, y_values2, 'r^', label='cos') 
#plt.xlim([0,2*np.pi])
#plt.legend()
#plt.show()
#
#• plot() 한번으로 여러 개의 좌표 찍기
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()
#plt.figure(figsize=(10,10))
#x = np.linspace(0, 10, 10) 
##plt.plot(x,x,'o',x,x**2,'^',x,x**3,'s')
#plt.plot(x,x,'rs', x,x**2,'g^', x,x**3,'b--')
##red squares, green triangles, blue dashes 
#plt.show()

#• 2개의 리스트로 그래프 그리기
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()
#plt.figure(figsize=(10,10))
#years = np.arange(2010, 2020)
#values = np.random.randint(500,2000,10) 
#plt.plot(years,values,'r^--') #red triangles solid line 
#plt.show()
##







