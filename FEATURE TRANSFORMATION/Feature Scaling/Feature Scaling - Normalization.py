# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 20:58:50 2023

@author: Justin Thomas
"""

'''
What is Normalization? 
It is a scaling technique method in which data points are shifted and rescaled so that they end up in a range of 0 to 1. 
It is also known as min-max scaling. The goal is to change the values of numeric columns in the dataset to use a common scale 

Types:-

1) Min - Max Scaling
2) Means Normalization
3) Max Absolute
4) Robust Scaling    

Standardization-------
Standardization (or Z-score normalization) is the process of rescaling the features so that they’ll have the properties of a Gaussian distribution with

μ=0 and σ=1
where μ is the mean and σ is the standard deviation from the mean; standard scores (also called z scores) 


Normalization------Mostly used for CNN or when we know min and max values
Normalization often also simply called Min-Max scaling basically shrinks the range of the data such that the range is
fixed between 0 and 1 (or -1 to 1 if there are negative values). It works better for cases in which the standardization 
might not work so well. If the distribution is not Gaussian or the standard deviation is very small, the min-max scaler works better.

Normalization is typically done via the following equation:

Use Cases----
Some examples of algorithms where feature scaling is important are:

1.K-nearest neighbors with a Euclidean distance measure if want all features to contribute equally.
2.Logistic regression, SVM, perceptrons, neural networks.
3.K-means.
4.Linear discriminant analysis, principal component analysis, kernel principal component analysis.


Drawbacks------
Normalizing the data is sensitive to outliers, so if there are outliers in the data set it is a bad practice.
Standardization creates a new data not bounded (unlike normalization).

NO feature scaling for Decision tree

But for outliers we use Robust Scaling in Normalization 
'''



import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('wine_data.csv',header=None,usecols=[0,1,2])
df.columns=['Class label', 'Alcohol', 'Malic acid']
df
sns.kdeplot(df['Alcohol'])
sns.kdeplot(df['Malic acid'])
color_dict={1:'red',3:'green',2:'blue'}
sns.scatterplot(df['Alcohol'],df['Malic acid'],hue=df['Class label'],palette=color_dict)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Class label', axis=1),df['Class label'],test_size=0.3,random_state=0)

X_train.shape, X_test.shape
X_train

##### SCALING

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled

# when we use sklearn classes it converts the data into a numpy array. so we connvert back it into a dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_train_scaled

np.round(X_train.describe(), 1)
np.round(X_train_scaled.describe(), 1)


################ VS Subplotting

## Scatterplot
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.scatter(X_train['Alcohol'], X_train['Malic acid'],c=y_train)
ax1.set_title("Before Scaling")
ax2.scatter(X_train_scaled['Alcohol'], X_train_scaled['Malic acid'],c=y_train)
ax2.set_title("After Scaling")
plt.show()


## Kernal probailit density kde

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['Alcohol'], ax=ax1)
sns.kdeplot(X_train['Malic acid'], ax=ax1)
# after scaling
ax2.set_title('After Standard Scaling')
sns.kdeplot(X_train_scaled['Alcohol'], ax=ax2)
sns.kdeplot(X_train_scaled['Malic acid'], ax=ax2)
plt.show()



# FOR ALCOHOL

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Alcohol Distribution Before Scaling')
sns.kdeplot(X_train['Alcohol'], ax=ax1)

# after scaling
ax2.set_title('Alcohol Distribution After Standard Scaling')
sns.kdeplot(X_train_scaled['Alcohol'], ax=ax2)
plt.show()



# FOR MALIC ACID

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Malic acid Distribution Before Scaling')
sns.kdeplot(X_train['Malic acid'], ax=ax1)

# after scaling
ax2.set_title('Malic acid Distribution After Standard Scaling')
sns.kdeplot(X_train_scaled['Malic acid'], ax=ax2)
plt.show()