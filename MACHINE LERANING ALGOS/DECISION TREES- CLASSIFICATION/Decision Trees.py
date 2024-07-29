# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 21:21:28 2023

@author: Justin Thomas
"""

import pandas as pd
from pandas_datareader import data
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV

boston = load_boston()
df = pd.DataFrame(boston.data)

df.columns = boston.feature_names
df['MEDV'] = boston.target

df.head()

X = df.iloc[:,0:13]
y = df.iloc[:,13]

X.shape
y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
rt = DecisionTreeRegressor(criterion = 'mse', max_depth=5)
rt.fit(X_train,y_train)

y_pred = rt.predict(X_test)
r2_score(y_test,y_pred)

########  Hyperparameter Tuning

param_grid = {
    'max_depth':[2,4,8,10,None],
    'criterion':['mse','mae'],
    'max_features':[0.25,0.5,1.0],
    'min_samples_split':[0.25,0.5,1.0]
}

reg = GridSearchCV(DecisionTreeRegressor(),param_grid=param_grid)
reg.fit(X_train,y_train)

GridSearchCV(cv=None, error_score=nan,
             estimator=DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse',
                                             max_depth=None, max_features=None,
                                             max_leaf_nodes=None,
                                             min_impurity_decrease=0.0,
                                             min_impurity_split=None,
                                             min_samples_leaf=1,
                                             min_samples_split=2,
                                             min_weight_fraction_leaf=0.0,
                                             presort='deprecated',
                                             random_state=None,
                                             splitter='best'),
             iid='deprecated', n_jobs=None,
             param_grid={'criterion': ['mse', 'mae'],
                         'max_depth': [2, 4, 8, 10, None],
                         'max_features': [0.25, 0.5, 1.0],
                         'min_samples_split': [0.25, 0.5, 1.0]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)


reg.best_score_
reg.best_params_

###### Feature Importance
for importance, name in sorted(zip(rt.feature_importances_, X_train.columns),reverse=True):
  print (name, importance)










