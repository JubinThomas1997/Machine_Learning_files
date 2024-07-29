# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 23:33:20 2023

@author: Justin Thomas
"""


'''
Binarization
Binarization is used to convert numerical feature vector to a binary vector. 
Binarization is a operation on count data, in which data scientist can decide to consider only the presence
 or absence of a characteristic rather than a quantified number of occurrences. Otherwise, 
it can be used as a pre-processing step for estimators that consider random Boolean values.
 It consists if-else condition in raw implementation.

suppose if we have data,

data = [ [3, -0.5, 2, 1],

[2.2, 3, 0, 1.4],

[3.1, 1.5, 0, 1] ]

If we apply binarizer on this with the threshold of 1.5. The matrix will be converted as.

[ [1, 0, 1, 0],

[1, 1, 0, 0],

[1, 1, 0, 1] ]

'''





import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.compose import ColumnTransformer

df = pd.read_csv('train.csv')[['Age','Fare','SibSp','Parch','Survived']]
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df.head()

df['family'] = df['SibSp'] + df['Parch']
df.head()

df.drop(columns=['SibSp','Parch'],inplace=True)
df.head()

X = df.drop(columns=['Survived'])
y = df['Survived']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.head()


# Without binarization

clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

accuracy_score(y_test,y_pred)   #0.6223776223776224
np.mean(cross_val_score(DecisionTreeClassifier(),X,y,cv=10,scoring='accuracy')) #0.64712441314554



# Applying Binarization

from sklearn.preprocessing import Binarizer

trf = ColumnTransformer([
    ('bin',Binarizer(copy=False),['family'])
],remainder='passthrough')

X_train_trf = trf.fit_transform(X_train)
X_test_trf = trf.transform(X_test)
pd.DataFrame(X_train_trf,columns=['family','Age','Fare'])

clf = DecisionTreeClassifier()
clf.fit(X_train_trf,y_train)
y_pred2 = clf.predict(X_test_trf)

accuracy_score(y_test,y_pred2)

X_trf = trf.fit_transform(X)
np.mean(cross_val_score(DecisionTreeClassifier(),X_trf,y,cv=10,scoring='accuracy'))
