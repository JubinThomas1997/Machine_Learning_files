# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 11:10:44 2023

@author: Justin Thomas
"""
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("placement.csv")
df
df.info()

df=df.iloc[:,1:]
df.head()
df.tail()

df.columns
plt.scatter(df.cgpa,df.iq,c=df.placement)


#Pre the data for train and test
df
X=df.iloc[:,0:2]            #Independand variables
y=df.iloc[:,-1]             #Dependand Variables
X
y


#For train test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
X_train


#To scale the data 
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
X_train=scalar.fit_transform(X_train)
X_train
X_test=scalar.transform(X_test)
X_test

#to train the model
from sklearn.linear_model import LogisticRegression
cls=LogisticRegression()
cls.fit(X_train, y_train)

#model evaluate
y_pred=cls.predict(X_test)
y_test

#acurracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

#decision boundary 
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_train, y_train.values, clf=cls, legend=2)  
