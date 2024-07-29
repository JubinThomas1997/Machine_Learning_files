import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import seaborn as sns

df = pd.read_csv('train.csv')[['Age','Pclass','SibSp','Parch','Survived']]
df.head()

df.dropna(inplace=True)
df.head()


X = df.iloc[:,0:4]
y = df.iloc[:,-1]
X.head()

np.mean(cross_val_score(LogisticRegression(),X,y,scoring='accuracy',cv=20)) #0.6933333333333332

#Applying Feature Construction
X['Family_size'] = X['SibSp'] + X['Parch'] + 1 # TO ADD THE PASSENGER ALSO
X.head()

def myfunc(num):
    if num == 1:
        #alone
        return 0
    elif num >1 and num <=4:
        # small family
        return 1
    else:
        # large family
        return 2

myfunc(4)

X['Family_type'] = X['Family_size'].apply(myfunc)
X.head()

X.drop(columns=['SibSp','Parch','Family_size'],inplace=True)
X.head()

#Feature Splitting
df = pd.read_csv('train.csv')
df.head()
df['Name']

df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df[['Title','Name']]

(df.groupby('Title').mean()['Survived']).sort_values(ascending=False)
df['Is_Married'] = 0
df['Is_Married'].loc[df['Title'] == 'Mrs'] = 1
df['Is_Married']
