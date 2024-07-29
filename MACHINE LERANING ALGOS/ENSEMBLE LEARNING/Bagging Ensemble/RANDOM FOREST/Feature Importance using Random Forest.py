# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:19:42 2023

@author: Justin Thomas
"""

import pandas as pd
import seaborn as sns
df = pd.read_csv('train.csv')
df.head()

X = df.iloc[:,1:]
y = df.iloc[:,0]
sns.heatmap(X.iloc[5].values.reshape(28,28))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X,y)

rf.feature_importances_

rf.feature_importances_.shape

sns.heatmap(rf.feature_importances_.reshape(28,28))