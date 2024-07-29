from sklearn.datasets import load_boston
import numpy as np

X,y=load_boston(return_X_y=True)
X
y

X.shape,y.shape


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score


lr=LinearRegression()
dt=DecisionTreeRegressor()
svr=SVR()

estimators=[('lr',lr),('dt',dt),('svr',svr)]

for estimator in estimators:
    scores= cross_val_score(estimator[1],X,y,scoring='r2',cv=10)
    print(estimator[0],np.round(np.mean(scores),2))


from sklearn.ensemble import VotingRegressor
vr = VotingRegressor(estimators)

for i in range(1,4):
  for j in range(1,4):
    for k in range(1,4):
        vr= VotingRegressor(estimators,weights=[i,j,k])
        scores = cross_val_score(vr,X,y,scoring='r2',cv=10)
        print("For i={},j={},k={}".format(i,j,k),np.round(np.mean(scores),2))
        
        


