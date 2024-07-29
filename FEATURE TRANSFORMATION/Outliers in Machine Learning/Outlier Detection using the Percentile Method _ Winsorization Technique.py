# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:41:49 2023

@author: Justin Thomas
"""

import numpy as np
import pandas as pd
df = pd.read_csv('weight-height.csv')
df.head()

df.shape
df['Height'].describe()

import seaborn as sns
sns.distplot(df['Height'])
sns.boxplot(df['Height'])

# trimming
upper_limit = df['Height'].quantile(0.99)   # keeping 99percetile as threshold for upper limit ,its on us 
upper_limit
lower_limit = df['Height'].quantile(0.01) # keeping 1percetile as threshold for lower limit
lower_limit

new_df = df[(df['Height'] <= 74.78) & (df['Height'] >= 58.13)]
new_df['Height'].describe()
sns.distplot(new_df['Height'])
sns.boxplot(new_df['Height'])

# Capping --> Winsorization
df['Height'] = np.where(df['Height'] >= upper_limit,
        upper_limit,
        np.where(df['Height'] <= lower_limit,
        lower_limit,
        df['Height']))

df.shape
df['Height'].describe()
sns.distplot(df['Height'])
