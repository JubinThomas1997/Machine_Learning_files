# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 08:47:25 2023

@author: Justin Thomas
"""
''' TO CONVERT COLUMNS WHICH HAVE MIXED DATA LIKE DA12 OR 123DR TO SEPERATE COULMS '''

import numpy as np
import pandas as pd
df = pd.read_csv('titanic.csv')
df.head()
df['number'].unique()

fig = df['number'].value_counts().plot.bar()
fig.set_title('Passengers travelling with')


# extract numerical part
df['number_numerical'] = pd.to_numeric(df["number"],errors='coerce',downcast='integer')
df['number_numerical'].isnull()
# extract categorical part
df['number_categorical'] = np.where(df['number_numerical'].isnull(),df['number'],np.nan) #to replace numerical in df['number_numerical'] with Nan

df.head()
df['Cabin'].unique()
df['Ticket'].unique()

df['cabin_num'] = df['Cabin'].str.extract('(\d+)') # captures numerical part
df['cabin_cat'] = df['Cabin'].str[0] # captures the first letter

df.head()
df['cabin_cat'].value_counts().plot(kind='bar')


# extract the last bit of ticket as number
df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])
df['ticket_num'] = pd.to_numeric(df['ticket_num'],errors='coerce',downcast='integer')

# extract the first part of ticket as category
df['ticket_cat'] = df['Ticket'].apply(lambda s: s.split()[0])
df['ticket_cat'] = np.where(df['ticket_cat'].str.isdigit(), np.nan,df['ticket_cat'])

df.head(20)
df['ticket_cat'].unique()
