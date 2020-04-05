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
                    nl = [sp[2],sp[3],sp[74],sp[12],sp[126],sp[16],sp[7],sp[10],sp[20],sp[22],sp[140],sp[24]]
                    line = ','.join(nl)
                fout.write(line+"\n")
    fout.close()

# convert_file()

# 
# Конвертируем и подготавливаем текстовый файл к работе
# 
def txt_to_csv(filename='train'):
    df       = pd.read_csv( filename+'2.txt',dtype='object' , sep=',', low_memory=False , index_col=False)
    df['dt'] = pd.to_datetime(df.Date+df.HrMn,format='%Y%m%d%H%M')
    df       = df.set_index('dt')
    df       = df.drop(['Date','HrMn'],axis=1)
    df       = df.astype('float')
    print(df.head())
    print(df.info(memory_usage='deep'))
    df.to_csv(filename+'.csv',sep=';')
# 
# 2. конвертируем текстовый файл в csv
# 
# txt_to_csv()

# 
# 3. Подготавливаем параметры для обучения
# 
def read_file(filename='train'):
    df = pd.read_csv(filename+'.csv', sep=';', low_memory=False , index_col='dt')
    return df

df = read_file()

# устанавливаем флаг тумана в отдельной колоке
def set_fog_flag(df=None):
    fog_values = [ 20,30,31,32,33,34,35 ]

    for index, row in df.loc[ (df.Visby>0) & (df.Visby<=1000) ].iterrows():
        if df.at[index, 'Wx'] in fog_values:
            df.at[index, 'fog'] = True
        else:
            df.at[index, 'fog'] = False
    # save file
    df.to_csv('train.csv',sep=';')

# set_fog_flag(df)


# 
# 4. cat boost train
# 
import catboost as cb
from catboost import CatBoostRegressor, Pool, CatBoostClassifier



X = np.array(df[['Cv','Hgt','Wx','Visby','Dir','Spd','Temp','Dewpt','RHx','Slp']].fillna(0))
y = np.array(df[['fog']].fillna(0),'uint8').reshape(-1)
# dataset = Pool(X, y)


print(X)
print(y)
train_data = Pool(data=X,
                  label=y)
print(train_data)
model = CatBoostClassifier(iterations=10)

model.fit(train_data)
# Get predicted classes
preds_class = model.predict(train_data)
print(preds_class)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(train_data)
print(preds_proba)
# Get predicted RawFormulaVal
preds_raw = model.predict(train_data, prediction_type='RawFormulaVal')
print(preds_raw)
exit()

# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations=2,
                          learning_rate=1,
                          depth=2)
# Fit model
model.fit(train_data, train_labels)
# Get predictions
preds = model.predict(eval_data)