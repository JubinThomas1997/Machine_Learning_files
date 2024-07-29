# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:45:57 2023

@author: Justin Thomas
"""

'''
Ordinal data is data that can be ranked or ordered. Examples include data taken from a poll or survey.(if there is a ordering like good>bad>baddest)

Nominal data is data that can be made to fit various categories. Examples include whether an animal is a mammal,
fish, reptile, amphibian, or bird



Ordinal encoder is used to encode datas in X colums or input labels
Label encoder is used to encode data in Y colums or output lables

'''

import numpy as np
import pandas as pd

df = pd.read_csv('customer.csv')
df.sample(5)

'''
HERE gender is a nominal data as it can be categorized, Age is numerical data,review is ordinal , and education is also
ordinal as it can be ordered , purchased is nominal

SO we used one hot encoder on nominal data
ordinal encoder on ordinal data 
and as output column is a nominal data we used Label encoding there

'''

df = df.iloc[:,2:]
df.head()

#Always do train test slipt before data transformation

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df.iloc[:,0:2],df.iloc[:,-1],test_size=0.2)
X_train

#Now encoding the data
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']]) # give prioties to data
# here we gave the max value to good and Pg thats why its witten at 3rd place and we made 2 colums because we are useingg-
#2 columns for ordinal encoding- review and education
oe.fit(X_train)
X_test= oe.transform(X_test)
X_train = oe.transform(X_train)

X_train    



# NOW using label encoding on output column, here we dont decide which data has the highest or lowest point like the ordinal encoding

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)
y_train
