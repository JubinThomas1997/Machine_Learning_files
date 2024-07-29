
# CCA or complete case analysis is done where the data is missing in random places

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_science_job.csv')
df

df.isnull().sum()
# to finde perc data missing
df.isnull().mean()*100
# here we can see gender has 23.530640% ,company_size has 30.994885% and company_type has 32.049274% data missing
# we can omit these colums because these are tooo big numbers and will cause problem later 

# we will work on the columns that has around 15% or less data missing 

df.shape

cols = [vals for vals in df.columns if df[vals].isnull().mean() < 0.05 and df[vals].isnull().mean() > 0]
cols
df[cols].sample(5)

df['education_level'].value_counts()

len(df[cols].dropna()) / len(df)  #0.8968577095730244 see even if we drop the colums we would have 89% data remaining
new_df = df[cols].dropna()
df.shape, new_df.shape  #((19158, 13), (17182, 5)) old and new data shape

new_df.hist(bins=50, density=True, figsize=(12, 12))
plt.show()


#### run together

#from the fig we can see both the old and new graphs are overlapping so no isues in droping the colums

fig = plt.figure()
ax = fig.add_subplot(111)
# original data
df['training_hours'].hist(bins=50, ax=ax, density=True, color='red')
# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
new_df['training_hours'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)


fig = plt.figure()
ax = fig.add_subplot(111)
# original data
df['training_hours'].plot.density(color='red')
# data after cca
new_df['training_hours'].plot.density(color='green')


fig = plt.figure()
ax = fig.add_subplot(111)
# original data
df['city_development_index'].hist(bins=50, ax=ax, density=True, color='red')
# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
new_df['city_development_index'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)


fig = plt.figure()
ax = fig.add_subplot(111)
# original data
df['city_development_index'].plot.density(color='red')
# data after cca
new_df['city_development_index'].plot.density(color='green')


fig = plt.figure()
ax = fig.add_subplot(111)
# original data
df['experience'].hist(bins=50, ax=ax, density=True, color='red')   #ax-axis
# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
new_df['experience'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)


fig = plt.figure()
ax = fig.add_subplot(111)
# original data
df['experience'].plot.density(color='red')
# data after cca
new_df['experience'].plot.density(color='green')


#####
 
########### for categorical colums



temp = pd.concat([# percentage of observations per category, original data
            df['enrolled_university'].value_counts() / len(df),
            # percentage of observations per category, cca data
            new_df['enrolled_university'].value_counts() / len(new_df)],axis=1)

# add column names
temp.columns = ['original', 'cca']

temp


temp = pd.concat([# percentage of observations per category, original data
            df['education_level'].value_counts() / len(df),
            # percentage of observations per category, cca data
            new_df['education_level'].value_counts() / len(new_df)],axis=1)

# add column names
temp.columns = ['original', 'cca']

temp


