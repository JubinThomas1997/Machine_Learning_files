    # -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 19:17:27 2023

@author: Justin Thomas
"""

import pandas as pd
import seaborn as sns

tips=sns.load_dataset('tips')
titanic=pd.read_csv('train.csv')
flights=sns.load_dataset('flights')
iris=sns.load_dataset('iris')

tips.head()
flights.head()
iris.head()
iris.sample(5)


############# 1. Scatterplot (Numerical - Numerical) #############
#Multivarient analysis-
sns.scatterplot(tips['total_bill'],tips['tip'],hue=tips['sex'],style=tips['smoker'],size=tips['size'])



############# 2. Bar Plot (Numerical - Categorical) #############
titanic.head()
sns.barplot(titanic.Survived,titanic.Pclass)

sns.barplot(titanic.Pclass,titanic.Fare)

sns.barplot(titanic.Pclass,titanic.Age,hue=titanic.Sex)


############# 3. Box Plot (Numerical - Categorical) #############

sns.boxplot(titanic['Sex'],titanic['Age'],hue=titanic['Survived'])
 

############# 4. Distplot (Numerical - Categorical) ############# probabillty
titanic.columns
sns.distplot(titanic[titanic['Survived']==0]['Age'],hist=False)
sns.distplot(titanic[titanic['Survived']==1]['Age'],hist=False)


############# 5. HeatMap (Categorical - Categorical) #############
pd.crosstab(titanic['Pclass'],titanic['Survived'])
sns.heatmap(pd.crosstab(titanic['Pclass'],titanic['Survived']))


(titanic.groupby('Pclass').mean()['Survived']*100)
(titanic.groupby('Embarked').mean()['Survived']*100)


############# 6. ClusterMap (Categorical - Categorical) #############
sns.clustermap(pd.crosstab(titanic['Parch'],titanic['Survived']))

############# 7. Pairplot #############
iris.head()
sns.pairplot(iris,hue='species')

############# 8. Lineplot (Numerical - Numerical) ############# used when x axis is a time varient quantity
flights.head()
new = flights.groupby('year').sum().reset_index()
sns.lineplot(new['year'],new['passengers'])
flights.pivot_table(values='passengers',index='month',columns='year')
sns.heatmap(flights.pivot_table(values='passengers',index='month',columns='year'))
sns.clustermap(flights.pivot_table(values='passengers',index='month',columns='year'))
