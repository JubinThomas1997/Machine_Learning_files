'''
IF BAGGING AND RF HAS BOTH DECISION TREES USED WHAT IS THE DIFFERNCE?
BAGGING TREES USES TREE LEVEL SAMPLING WHERE AS RANDOM FORESTS USES NODE LEVEL SAMPLING, HENCE THERE WOULD BE
MORE RANDOMNESS IN THE TREES AND LEADS TO LESS VARIACES
'''

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

X,y = make_classification(n_features=5, n_redundant=0, n_informative=5,n_clusters_per_class=1)

df = pd.DataFrame(X,columns=['col1','col2','col3','col4','col5'])
df['target'] = y
print(df.shape)
df.head()

bag = BaggingClassifier(max_features=2)
bag.fit(df.iloc[:,:5],df.iloc[:,-1])

plt.figure(figsize=(12,12))
plot_tree(bag.estimators_[0])
plt.show()


rf = RandomForestClassifier(max_features=2)
rf.fit(df.iloc[:,:5],df.iloc[:,-1])

plt.figure(figsize=(12,12))
plot_tree(rf.estimators_[4])
plt.show()