# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:38:43 2023

@author: Justin Thomas
"""

#1. Importing pandas
import pandas as pd


#2. Opening a local csv file
df = pd.read_csv('aug_train.csv')
df


#3. Opening a csv file from an URL
import requests
from io import StringIO

url = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0"}
req = requests.get(url, headers=headers)
data = StringIO(req.text)

pd.read_csv(data)


# 4. Sep Parameter
df1=pd.read_csv('movie_titles_metadata.tsv.txt',sep='\t',names=['sno','name','release_year','rating','votes','genres'])
df1
df1.columns


# 5. Index_col parameter
pd.read_csv('aug_train.csv',index_col='enrollee_id')

#6. Header parameter
df2=pd.read_csv('test.csv',header=1)
df2


#7. use_cols parameter
pd.read_csv('aug_train.csv',usecols=['enrollee_id','gender','education_level'])


#9. Skiprows/nrows Parameter- shows the only required rows
pd.read_csv('aug_train.csv',nrows=100)

#10.to analysis 2 colums
pd.crosstab(data['Pclass'],data['Survived'])