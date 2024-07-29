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
sns.scatterplot(tips['total_bill'],tips['tip'],hue=tips['sex'],style=tips['size'],size=tips['smoker'])


############# 2. Bar Plot (Numerical - Categorical) #############
sns.barplot(tips['total_bill'], tips['sex'],hue=tips['time'])
sns.barplot(titanic.Survived,titanic.Pclass)

sns.barplot(titanic.Pclass,titanic.Fare)

sns.barplot(titanic.Pclass,titanic.Age,hue=titanic.Sex)

sns.boxplot(titanic['Sex'],titanic['Age'],hue=titanic['Survived'])
