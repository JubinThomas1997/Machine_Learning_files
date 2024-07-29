# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 20:24:09 2023

@author: Justin Thomas
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('train.csv')
df.head()


##### Let's Plan

df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)

# Step 1 -> train/test/split
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']),df['Survived'],test_size=0.2,random_state=42)

X_train.head()
y_train.sample(5)

# imputation transformer
trf1 = ColumnTransformer([
    ('impute_age',SimpleImputer(),[2]),
    ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])
],remainder='passthrough')

# one hot encoding
trf2 = ColumnTransformer([
    ('ohe_sex_embarked',OneHotEncoder(sparse=False,handle_unknown='ignore'),[1,6])
],remainder='passthrough')

# Scaling
trf3 = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,10))
]) # 0-10 because there are 7 coulms and out of it 2 will become 5 because of one hot encoding and due to dropping of 1 coul from each 2 colums we are left with 7+3=10


# Feature selection
trf4 = SelectKBest(score_func=chi2,k=8)

# train the model
trf5 = DecisionTreeClassifier()

##### Create Pipeline
pipe = Pipeline([
    ('trf1',trf1),
    ('trf2',trf2),
    ('trf3',trf3),
    ('trf4',trf4),
    ('trf5',trf5)
])


# train
pipe.fit(X_train,y_train)

######## Explore the Pipeline

# Code here
pipe.named_steps

# Display Pipeline
from sklearn import set_config
set_config(display='diagram')

# Predict
y_pred = pipe.predict(X_test)

y_pred

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

# cross validation using cross_val_score
from sklearn.model_selection import cross_val_score
cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()