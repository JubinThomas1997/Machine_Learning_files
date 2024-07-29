# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 21:54:22 2023

@author: Justin Thomas
"""

# used when data is skewed
#skewed can be found by using distplot and [.skew()-if it is really close TO 0 THEN NORMAL DISTRIBUTED  if it for from 0 then 
#right skewed and if it in large negative numbers then left skewed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('placement.csv')
df.head()


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['cgpa'])

plt.subplot(1,2,2)
sns.distplot(df['placement_exam_marks'])

plt.show()
#here we can see cgpa is normally distributed but placemnet is a bit rightly skewed
''' 
Normal distribution, also known as the Gaussian distribution, 
is a probability distribution that is symmetric about the mean, 
showing that data near the mean are more frequent in occurrence than data far from the mean.
'''

df['placement_exam_marks'].skew()  # 0.8356419499466834 ao rightly skewed
df['cgpa'].skew() # -0.014529938929314918 not exactly 0 but close enough so normally distributed

df['placement_exam_marks'].describe()
sns.boxplot(df['placement_exam_marks']) # we can see so many outliners

# Finding the IQR
percentile25 = df['placement_exam_marks'].quantile(0.25)
percentile75 = df['placement_exam_marks'].quantile(0.75)
percentile75

iqr = percentile75 - percentile25
iqr

'''
The IQR may also be called the midspread, middle 50%, fourth spread, or H‑spread. 
It is defined as the difference between the 75th and 25th percentiles of the data.
To calculate the IQR, the data set is divided into quartiles, 
or four rank-ordered even parts via linear interpolation. These quartiles are denoted by Q1 
(also called the lower quartile), Q2 (the median), and Q3 (also called the upper quartile). 
The lower quartile corresponds with the 25th percentile and the upper quartile corresponds 
with the 75th percentile, 
so IQR = Q3 −  Q1[1]. 
it is the blue box in the boxplot
'''

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print("Upper limit",upper_limit)
print("Lower limit",lower_limit)

'''
 If the distribution is standard normal the IQR is about 1.35 so 1.5 times that is 2.025
 so the area beyond a point that far from the mean is about 2.5%.
 But the inner fences of a box plot are 1.5 times the IQR from the quartiles
'''


#Finding Outliers
df[df['placement_exam_marks'] > upper_limit]
df[df['placement_exam_marks'] < lower_limit]



######## Trimming
new_df = df[df['placement_exam_marks'] < upper_limit]
new_df.shape

# Comparing
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(2,2,3)
sns.distplot(new_df['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df['placement_exam_marks'])

plt.show()

# Capping
new_df_cap = df.copy()

new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
)

new_df_cap.shape

# Comparing
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(2,2,3)
sns.distplot(new_df_cap['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df_cap['placement_exam_marks'])

plt.show()