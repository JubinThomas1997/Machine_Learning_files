import numpy as np
from sklearn.datasets import load_diabetes


X,y = load_diabetes(return_X_y=True)
X
y

X.shape
y.shape


##############  Using Sklearn's Linear Regression

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

reg.coef_ # m 
reg.intercept_ #b

###################### Making our own Linear Regression Class

class MeraLR:
    
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self,X_train,y_train):
        X_train = np.insert(X_train,0,1,axis=1)  # to add the 1's in the x matrix -- to the '0'th index,'1' as in 1 to put , axis=1 is column
        
        # calcuate the coeffs
        betas = np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        # np.linalg.inv calculate sthe ineverse
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
    
    def predict(self,X_test):
        y_pred = np.dot(X_test,self.coef_) + self.intercept_
        return y_pred
    
lr = MeraLR()
lr.fit(X_train,y_train)
X_train.shape

np.insert(X_train,0,1,axis=1).shape

y_pred = lr.predict(X_test)
r2_score(y_test,y_pred)

lr.coef_
lr.intercept_
