import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df=pd.read_csv("placement.csv")
df.head()                           #gives an overview of data frame
df.shape

df.info()


# Steps

# 0. Preprocess + EDA + Feature Selection
# 1. Extract input and output cols
# 2. Scale the values
# 3. Train test split
# 4. Train the model
# 5. Evaluate the model/model selection
# 6. Deploy the model


# 0.  PRE Processing 
# here we are going to remove the 1st column

df=df.iloc[:,1:]
df.head
# or use df=pd.read_csv("placement.csv",usecols=['cgpa','iq','placement'])

plt.scatter(df['cgpa'], df['iq'],c=df['placement'])


# 1.Extract input and output cols

x=df.iloc[:,0:2]  #to capture all the independ variables
y=df.iloc[:,-1]   #to capture the dependand variables


# 3. Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)  # out of 100- 10 rows for test, 90 for train
x_train
y_train


# 2. Scale the values
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_train

x_test=scalar.transform(x_test)    #we didn't use fit_ because we learned the pattern during the training
x_test


# 4. Train the model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(x_train,y_train)    # model training

y_pred = clf.predict(x_test)
y_pred
y_test

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x_train, y_train.values, clf=clf, legend=2)   # we converted the y_train into np array because we alreday scalled and made x train to np array using scalling

