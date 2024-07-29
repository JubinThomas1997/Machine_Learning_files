# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:07:55 2023

@author: Justin Thomas
"""

import pandas as pd


df=pd.read_csv('train.csv')
df

#1 . how big is the data
df.shape

#2.How does the data look like
df.head()
df.sample(5)  #gives 5 random samples


#3. What is the datatype of the give data
df.info()

#4. Chceking for any missing values
df.isnull().sum()

#5. How does the data look mathematically
df.describe()

#6. Are there any duplicate datas
df.duplicated().sum()

#7. How is the corelation between data
df.corr()
df.corr()['Survived']

