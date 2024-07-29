import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification

X,y = make_classification(n_features=5, n_redundant=0, n_informative=5,n_clusters_per_class=1) # default 100 samples

df = pd.DataFrame(X,columns=['col1','col2','col3','col4','col5'])
df['target'] = y

print(df.shape)
df.head()
df.columns

# function for row sampling

def sample_rows(df,percent):
  return df.sample(int(percent*df.shape[0]),replace=True)  #The sample() function is used to get a random sample of items from an axis of object

# function for feature sampling
def sample_features(df,percent):
    cols=random.sample(df.columns.tolist()[:-1],int(percent*(df.shape[1]-1))) #-1 for colums target
    new_df=df[cols]
    new_df['target']=df['target']
    return new_df

# function for combined sampling

def combined_sampling(df,row_percent,col_percent):
  new_df = sample_rows(df,row_percent)
  return sample_features(new_df,col_percent)


############### Code for ROW SAMPLING ###############
df1=sample_rows(df,0.1)
df2=sample_rows(df, 0.1)
df3=sample_rows(df, 0.1)

df1.shape

from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier()
clf2 = DecisionTreeClassifier()
clf3 = DecisionTreeClassifier()

clf1.fit(df1.iloc[:,0:5],df1.iloc[:,-1])
clf2.fit(df2.iloc[:,0:5],df2.iloc[:,-1])
clf3.fit(df3.iloc[:,0:5],df3.iloc[:,-1])

from sklearn.tree import plot_tree
plot_tree(clf1)
plot_tree(clf2)
plot_tree(clf3)

clf1.predict(np.array([ 1.190428,  0.781112, -1.751515,  1.034136,  0.275110 ]).reshape(1,5)) #0
clf2.predict(np.array([ 1.190428,  0.781112, -1.751515,  1.034136,  0.275110 ]).reshape(1,5)) #0
clf3.predict(np.array([ 1.190428,  0.781112, -1.751515,  1.034136,  0.275110 ]).reshape(1,5)) #0


clf1.predict(np.array([ -1.013017, -0.160818,  1.529460,  2.176947,  0.407910 ]).reshape(1,5)) #gives 1 
clf2.predict(np.array([ -1.013017, -0.160818,  1.529460,  2.176947,  0.407910 ]).reshape(1,5)) #gives 1
clf3.predict(np.array([ -1.013017, -0.160818,  1.529460,  2.176947,  0.407910 ]).reshape(1,5)) #gives 1


############### Code for COLUMN SAMPLING ###############
df1=sample_features(df,0.8)   #0.8*5=4 columns
df2=sample_features(df,0.8)
df3=sample_features(df,0.8)

df1.shape

from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier()
clf2 = DecisionTreeClassifier()
clf3 = DecisionTreeClassifier()

clf1.fit(df1.iloc[:,0:5],df1.iloc[:,-1])
clf2.fit(df2.iloc[:,0:5],df2.iloc[:,-1])
clf3.fit(df3.iloc[:,0:5],df3.iloc[:,-1])

from sklearn.tree import plot_tree
plot_tree(clf1)
plot_tree(clf2)
plot_tree(clf3)

clf1.predict(np.array([ 1.190428,  0.781112, -1.751515,  1.034136,  0.275110 ]).reshape(1,5)) #0
clf2.predict(np.array([ 1.190428,  0.781112, -1.751515,  1.034136,  0.275110 ]).reshape(1,5)) #0
clf3.predict(np.array([ 1.190428,  0.781112, -1.751515,  1.034136,  0.275110 ]).reshape(1,5)) #0


clf1.predict(np.array([ -1.013017, -0.160818,  1.529460,  2.176947,  0.407910 ]).reshape(1,5)) #gives 1 
clf2.predict(np.array([ -1.013017, -0.160818,  1.529460,  2.176947,  0.407910 ]).reshape(1,5)) #gives 1
clf3.predict(np.array([ -1.013017, -0.160818,  1.529460,  2.176947,  0.407910 ]).reshape(1,5)) #gives 1



