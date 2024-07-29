# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:17:47 2023

@author: Justin Thomas
"""

#we will use titanic dataset to learn EDA using univarient analysis


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('train.csv')
df

df.isnull().sum()
df.corr()

sns.heatmap(df.corr())


########## 1. Categorical Data

#a. Countplot

sns.countplot(df.Survived)
sns.countplot(df['Embarked'])
sns.countplot(df.Pclass)

#df['Survived'].value_counts().plot(kind='bar')

#b. PieChart
df['Sex'].value_counts().plot(kind='pie',autopct='%.2f')


########### 2. Numerical Data

#a. Histogram

plt.hist(df['Age'],bins=5)

#b. Distplot
sns.distplot(df['Age'])

sns.boxplot(df.Age)

df.Age.min()
df.Age.max()
df.Age.mean()
df.Age.skew()

df.Age.skew()
