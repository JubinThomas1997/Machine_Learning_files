# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:20:33 2023

@author: Justin Thomas
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('train.csv')
df.head()

df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
df.head()


# Step 1 -> train/test/split
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']),df['Survived'],test_size=0.2,random_state=42)
X_train.head(2)
y_train.head()

df.isnull().sum()


# Applying imputation

si_age = SimpleImputer()            #this will impute the mean of the age data
X_train_age = si_age.fit_transform(X_train[['Age']])     # [[]] because it is a 2D array
X_test_age = si_age.transform(X_test[['Age']])        
                    
si_embarked = SimpleImputer(strategy='most_frequent')       #this will impute the most frequent value of the age data
X_train_embarked = si_embarked.fit_transform(X_train[['Embarked']])
X_test_embarked = si_embarked.transform(X_test[['Embarked']])
X_train_embarked


# one hot encoding Sex and Embarked

ohe_sex = OneHotEncoder(sparse=False,handle_unknown='ignore')
X_train_sex = ohe_sex.fit_transform(X_train[['Sex']])
X_test_sex = ohe_sex.transform(X_test[['Sex']])

ohe_embarked = OneHotEncoder(sparse=False,handle_unknown='ignore')
X_train_embarked = ohe_embarked.fit_transform(X_train_embarked)
X_test_embarked = ohe_embarked.transform(X_test_embarked)


X_train_rem = X_train.drop(columns=['Sex','Age','Embarked'])
X_test_rem = X_test.drop(columns=['Sex','Age','Embarked'])

X_train_transformed = np.concatenate((X_train_rem,X_train_age,X_train_sex,X_train_embarked),axis=1)
X_test_transformed = np.concatenate((X_test_rem,X_test_age,X_test_sex,X_test_embarked),axis=1)

X_train_embarked
X_train_transformed
X_train.head(2)

clf = DecisionTreeClassifier()
clf.fit(X_train_transformed,y_train)
y_pred = clf.predict(X_test_transformed)
y_pred

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
