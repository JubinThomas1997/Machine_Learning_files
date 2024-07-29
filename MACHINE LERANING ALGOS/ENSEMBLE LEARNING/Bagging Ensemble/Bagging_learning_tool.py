# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:47:34 2023

@author: Justin Thomas
"""

############ RUN ON KAGELL  ###########

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_csv('/kaggle/input/iris/Iris.csv')
df.head()

df = df.iloc[:,1:]
df.head()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])
df.head()

df = df[df['Species'] != 0][['SepalWidthCm','PetalLengthCm','Species']]
df.head()


import seaborn as sns
import matplotlib.pyplot as plt

plt.scatter(df['SepalWidthCm'],df['PetalLengthCm'],c=df['Species'],cmap='winter')

df_train = df.iloc[:60,:].sample(10)
df_train

# Taking only 10 rows for training
df = df.sample(100)
df_train = df.iloc[:60,:].sample(10)
df_val = df.iloc[60:80,:].sample(5)
df_test = df.iloc[80:,:].sample(5)

df_train
df_val
df_test

X_test = df_val.iloc[:,0:2].values
y_test = df_val.iloc[:,-1].values
y_test


######################### Case 1 - Bagging ########################


############ Data for Tree 1

df_bag = df_train.sample(8,replace=True)

X = df_bag.iloc[:,0:2]
y = df_bag.iloc[:,-1]

df_bag

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score


# here we have not specified any max depth to decisions tree,so it will take the depth as maximun and result in low bias and high variance
dt_bag1 = DecisionTreeClassifier()  
evaluate(dt_bag1,X,y)

############## Data for Tree 2
df_bag = df_train.sample(8,replace=True)   #in replace one row can be repeated multiple times

# Fetch X and y
X = df_bag.iloc[:,0:2]
y = df_bag.iloc[:,-1]

# print df_bag
df_bag

dt_bag2 = DecisionTreeClassifier()
evaluate(dt_bag2,X,y)

############### Data for Tree 3
df_bag = df_train.sample(8,replace=True)

# Fetch X and y
X = df_bag.iloc[:,0:2]
y = df_bag.iloc[:,-1]

# print df_bag
df_bag

dt_bag3 = DecisionTreeClassifier()
evaluate(dt_bag3,X,y)


def evaluate(clf,X,y):
    clf.fit(X,y)
    plot_tree(clf)
    plt.show()
    plot_decision_regions(X.values, y.values, clf=clf, legend=2)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test,y_pred))



#### Predict

df_test

print("Predictor 1",dt_bag1.predict(np.array([2.2,5.0]).reshape(1,2))) #using values of df.test
print("Predictor 2",dt_bag2.predict(np.array([2.2,5.0]).reshape(1,2)))
print("Predictor 3",dt_bag3.predict(np.array([2.2,5.0]).reshape(1,2)))

### Pasting
# Row sampling without replacement
df_train
df_train.sample(8)

### Random Subspaces (here we do comulmn sampling)
df1 = pd.read_csv('/kaggle/input/iris/Iris.csv')
df1 = df1.sample(10)

df1
df1.sample(2,replace=True,axis=1)

#### Random Patches (Both row and column sampling)
df1
df1.sample(8,replace=True).sample(2,replace=True,axis=1)







