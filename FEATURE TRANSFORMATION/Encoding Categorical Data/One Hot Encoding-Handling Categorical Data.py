# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 11:51:55 2023

@author: Justin Thomas
"""

'''
WE convert the string into vectors
we also remove 1 column which is know as dummpy variable due to multicolinearity

'''

import numpy as np
import pandas as pd
df = pd.read_csv('cars.csv')
df.head()

df['brand'].value_counts()
df['brand'].nunique()

df['owner'].value_counts()
df['owner'].nunique()

df['fuel'].value_counts()
df['fuel'].nunique()



############################ for Ml we dont use OHE VIA PANDAS#####################
###1. OneHotEncoding using Pandas
pd.get_dummies(df,columns=['fuel','owner'])


#### 2. K-1 OneHotEncoding  - For multico-linearity
pd.get_dummies(df,columns=['fuel','owner'],drop_first=True)
#######################################################################################



##############3. OneHotEncoding using Sklearn

df.iloc[:,0:4]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,0:4],df.iloc[:,-1],test_size=0.2,random_state=2)

X_train.head()
y_train

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first',sparse=False,dtype=np.int32)           # here we drop the first colum as dummy trap
X_train_new = ohe.fit_transform(X_train[['fuel','owner']])
X_train_new

# OneHotEncoding with Top Categories

counts = df['brand'].value_counts()
counts
df['brand'].nunique()
threshold = 100

repl =counts[counts <= threshold].index  #.index used to make the column
repl
pd.get_dummies(df['brand'].replace(repl, 'uncommon')).sample(5)