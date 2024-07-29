# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:25:35 2023

@author: Justin Thomas
"""

################# KNN ###################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Social_Network_Ads.csv')
data

data.iloc[:,2:4]

X=data.iloc[:,2:4].values # to convert it into nump array
X
X.shape

y=data.iloc[:,-1].values
y
y.shape

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
X_train.shape


from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scalar.fit(X_train)
X_train=scalar.transform(X_train)
X_test=scalar.transform(X_test)

X_train
X_test


## 1.st method
X_train.shape
np.sqrt(X_train.shape[0])

# so assuming value of k as 17
k=17

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=k)

#train our model
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
y_pred.shape

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# 2nd method

accuracy=[]
for i in range(1,26):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    accuracy.append(accuracy_score(y_test,knn.predict(X_test)))
    
len(accuracy)
plt.plot(range(1,26),accuracy)

#so from fig we can see high acc is at 5
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
knn.predict(X_test)
accuracy_score(y_test,y_pred)



################ lets create a function that can show us the output if we provide a new input

def pred_out():
    age=int(input("Enter the age : "))
    salary=int(input("Enter the salary : "))
    
    X_new=np.array([[age],[salary]]).reshape(1,2)  #becauase X_train has a shape of(_,2)
    X_new=scalar.transform(X_new)
    
    if knn.predict(X_new)[0]==0:
        return "Will not purchase."
    else:
        return "Will purchase."

pred_out()


##################### FEW OBSERVATIONS ###############################


# Creating a sample Mesh grid

x= np.array([1,2,3])
y=np.array([4,5,6,7])

XX,YY=np.meshgrid(x,y)
XX.shape
YY.shape

# PLoting a function using meshgrid

x= np.linspace(-40,40,100)
y=np.linspace(-50,50,90)
XX,YY=np.meshgrid(x,y)
Z=(XX**2) + (YY**2)   # formula of elips
plt.contourf(XX,YY,Z)


x= np.linspace(-40,40,100)
y=np.linspace(-50,50,90)
XX,YY=np.meshgrid(x,y)
Z=np.random.random((90,100))     # formula of random
plt.contourf(XX,YY,Z)
#############################################################################

                            # MAKING A DECISION BOUNDARY #

#### Create a mesggrid

a=np.arange(start=X_train[:,0].min()-1,stop=X_train[:,0].max()+1,step=0.01)
b=np.arange(start=X_train[:,1].min()-1,stop=X_train[:,1].max()+1,step=0.01)
a
a.shape
b.shape
XX,YY=np.meshgrid(a,b)
XX.shape

# Classify every point on the meshgrid
'''
print(XX[0][0])
print(YY[0][0])
knn.predict(np.array([-2.6143637028796842,-2.818791604446135]).reshape(1,2))'''

np.array([XX.ravel(),YY.ravel()]).shape                 # ravel transforms higher dimension array into 1D array

"""
example for ravel
m=np.array([1,2,3],[4,5,6])
n=np.array([7,8,9],[0,0,0])
np.array([m.ravel(),n.ravel()]).shape
"""
input_array=np.array([XX.ravel(),YY.ravel()]).T
labels=knn.predict(input_array)

#ploting the array as image

plt.contourf(XX,YY,labels.reshape(XX.shape))


#plotting all the training data on the plot
plt.contourf(XX,YY,labels.reshape(XX.shape),alpha=0.75)
plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
