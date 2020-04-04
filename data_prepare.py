# -*- coding: utf-8 -*-
import csv
from datetime import datetime, timedelta

# то, что нужно для монги
import os.path
import os
import json
import numpy as np
import pandas as pd

#
# многопроцессорность
#
import multiprocessing, logging
from multiprocessing import Process, Pool

# Loading the .txt file
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# нулевой этап обрабтываем файл
import re


def convert_file():
    # fin = open("train.txt", "rt")
    fout = open("train2.txt", "wt")

    count = 0
    print("\nUsing readline()") 
    with open("train.txt") as fp: 
        while True: 
            line = fp.readline() 
            if not line:
                break

            if count==1:
                line=re.sub('\s+',',',line)
            count += 1
            if count>1:
                line=re.sub('\s+','',line)
                sp = line.split(',')
                # 'Date','HrMn','Cv','Hgt','Wx.17','Visby','Dir','Spd','Temp','Dewpt','RHx','Slp'
                if len(sp)>=140:
                    nl = [sp[2],sp[3],sp[74],sp[12],sp[126],sp[16],sp[7],sp[20],sp[22],sp[140],sp[24]]
                    line = ','.join(nl)
                fout.write(line+"\n")
    

    # first = False
    # i=0
    # for line in fin:
    # 	if i>0:
    #         l=line
    #         if i==2:
    #             l=re.sub('\s+',',',line)
    #         # fout.write(re.sub('\s+',' ',l))
    #         print(l.split(','))
    #     i++
        
    # fin.close()
    fout.close()

# exit()
# use_cols = ['Date','HrMn','Cv','Hgt','Wx.17','Visby','Dir','Spd','Temp','Dewpt','RHx','Slp']
# # первый этап - читаем файл
# data_iterator = pd.read_csv('train2.txt' , sep=',', usecols=use_cols, low_memory=False , index_col=False, chunksize=10000)

# chunk_list = []  

# Each chunk is in dataframe format
# for data_chunk in data_iterator:  
#     filtered_chunk = chunk_filtering(data_chunk)
#     print(data_chunk)
#     chunk_list.append(filtered_chunk)
#     exit()
    
# filtered_data = pd.concat(chunk_list)
    
# concat the list into dataframe 
# df_concat = pd.concat(chunk_list)
# print(df_concat.head())
# df_concat.to_csv('train_1_step.csv')
# df = pd.read_csv('train2.txt' , sep=',',dtype='string', low_memory=False , index_col=False)
# df = df[['Date','HrMn','Cv','Hgt','Wx.17','Visby','Dir','Spd','Temp','Dewpt','RHx','Slp']]
# df.to_csv('train_1_step.csv')
# print(df.head())

# df = df.astype('string', copy=False)
# print(df.iloc[10])
# print(df.columns.tolist())