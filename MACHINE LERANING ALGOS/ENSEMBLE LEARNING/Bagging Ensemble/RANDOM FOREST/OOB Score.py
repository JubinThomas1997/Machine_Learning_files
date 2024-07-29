# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:45:04 2023

@author: Justin Thomas
"""

#Out of bag (OOB) score is a way of validating the Random forest model. 
#Below is a simple intuition of how is it calculated followed by a ...

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

df = pd.read_csv('heart.csv')
df.head()

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

rf = RandomForestClassifier(oob_score=True)
rf.fit(X_train,y_train)

rf.oob_score_

y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred)