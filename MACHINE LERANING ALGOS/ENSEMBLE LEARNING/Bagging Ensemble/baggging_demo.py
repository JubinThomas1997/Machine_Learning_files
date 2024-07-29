# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:36:27 2023

@author: Justin Thomas
"""

" in bagging we use same models , where as in voling we use differnt models and use its total power"


from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


X,y = make_classification(n_samples=10000, n_features=10,n_informative=3)  #n_samples=rows required ,n_features=columns

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

print("Decision Tree accuracy",accuracy_score(y_test,y_pred))


### Bagging

bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=500,     #The number of base estimators in the ensemble
    max_samples=0.5,       # 0.5 of all the rows
    bootstrap=True,    #bootstrap true means row sampling with replacement
    random_state=42
)

bag.fit(X_train,y_train)

y_pred = bag.predict(X_test)
accuracy_score(y_test,y_pred)

bag.estimators_samples_[0].shape

bag.estimators_features_[0].shape

#### Bagging using SVM

bag = BaggingClassifier(
    base_estimator=SVC(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=True,
    random_state=42
)

bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
print("Bagging using SVM",accuracy_score(y_test,y_pred))

### Pasting
bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=False,    #bootstrap Fasle means rwo sampling without replacement
    random_state=42,
    verbose = 1,
    n_jobs=-1
)

bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
print("Pasting classifier",accuracy_score(y_test,y_pred))


#### Random Subspaces

bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=1.0,
    bootstrap=False,
    max_features=0.5,
    bootstrap_features=True,  #bootstrap_features means column sampling 
    random_state=42
)

bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
print("Random Subspaces classifier",accuracy_score(y_test,y_pred))

bag.estimators_samples_[0].shape

bag.estimators_features_[0].shape

#### Random Patches
bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=True,
    max_features=0.5,
    bootstrap_features=True,
    random_state=42
)

bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
print("Random Patches classifier",accuracy_score(y_test,y_pred))


### OOB Score
bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=True,
    oob_score=True,
    random_state=42
)

bag.fit(X_train,y_train)

bag.oob_score_

y_pred = bag.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))


'''
                                 Bagging Tips
Bagging generally gives better results than Pasting
Good results come around the 25% to 50% row sampling mark
Random patches and subspaces should be used while dealing with high dimensional data
To find the correct hyperparameter values we can do GridSearchCV/RandomSearchCV
'''

### Applying GridSearchCV
from sklearn.model_selection import GridSearchCV

parameters = {
    'n_estimators': [50,100,500], 
    'max_samples': [0.1,0.4,0.7,1.0],
    'bootstrap' : [True,False],
    'max_features' : [0.1,0.4,0.7,1.0]
    }

search = GridSearchCV(BaggingClassifier(), parameters, cv=5)
search.fit(X_train,y_train)

GridSearchCV(cv=5, error_score=nan,
             estimator=BaggingClassifier(base_estimator=None, bootstrap=True,
                                         bootstrap_features=False,
                                         max_features=1.0, max_samples=1.0,
                                         n_estimators=10, n_jobs=None,
                                         oob_score=False, random_state=None,
                                         verbose=0, warm_start=False),
             iid='deprecated', n_jobs=None,
             param_grid={'bootstrap': [True, False],
                         'max_features': [0.1, 0.4, 0.7, 1.0],
                         'max_samples': [0.1, 0.4, 0.7, 1.0],
                         'n_estimators': [50, 100, 500]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)

search.best_params_
search.best_score_

search.best_params_

