# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:23:49 2023

@author: Justin Thomas
"""

import numpy as np
import pandas as pd

date = pd.read_csv('orders.csv')
time = pd.read_csv('messages.csv')

date.head()
time.head()

date.info()  # HERE WE CAN SEE THE Dtype IS object SO WE NNED TO CONVERT IT SO THAT WE CAN WORK ON IT
time.info()     # HERE WE CAN SEE THE Dtype IS object SO WE NNED TO CONVERT IT SO THAT WE CAN WORK ON IT


#### Working with Dates

# Converting to datetime datatype
date['date'] = pd.to_datetime(date['date'])   # Dtype - datetime64[ns]
date.info()

#1. Extract year

date['date_year'] = date['date'].dt.year
date.sample(5)

#2. Extract Month

date['date_month_no'] = date['date'].dt.month
date.head()

# MONTH NAME
date['date_month_name'] = date['date'].dt.month_name()
date.head()

#3. Extract Days
date['date_day'] = date['date'].dt.day
date.head()

#4. day of week
date['date_dow'] = date['date'].dt.dayofweek
date.head()

#5. day of week - name
date['date_dow_name'] = date['date'].dt.day_name()
date.drop(columns=['product_id','city_id','orders']).head()

#6. is weekend?

date['date_is_weekend'] = np.where(date['date_dow_name'].isin(['Sunday', 'Saturday']), 1,0)
date.drop(columns=['product_id','city_id','orders']).head()


#7.Extract week of the year
date['date_week'] = date['date'].dt.week
date.drop(columns=['product_id','city_id','orders']).head()

#8.Extract Quarter
date['quarter'] = date['date'].dt.quarter
date.drop(columns=['product_id','city_id','orders']).head()

#9.Extract Semester - 2 SEM IN A YEAR
date['semester'] = np.where(date['quarter'].isin([1,2]), 1, 2)
date.drop(columns=['product_id','city_id','orders']).head()


#10. Extract Time elapsed between dates
import datetime
today = datetime.datetime.today()
today
today - date['date']
(today - date['date']).dt.days #FOR DAYS

# Months passed
np.round((today -date['date']) / np.timedelta64(1, 'M'),0)



##### FOR TIME

time.info()

# Converting to datetime datatype
time['date'] = pd.to_datetime(time['date'])
time.info()

time['hour'] = time['date'].dt.hour
time['min'] = time['date'].dt.minute
time['sec'] = time['date'].dt.second

time.head()

#Extract Time part
time['time'] = time['date'].dt.time
time.head()

#Time difference
today - time['date']

# in seconds
(today - time['date'])/np.timedelta64(1,'s')

# in minutes
(today - time['date'])/np.timedelta64(1,'m')

# in hours
(today - time['date'])/np.timedelta64(1,'h')



