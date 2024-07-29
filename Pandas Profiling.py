# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:39:22 2023

@author: Justin Thomas
"""

import pandas as pd
import numpyas np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('train.csv')

df.head()

from pandas_profiling import ProfileReport
prof=ProfileReport(df)
prof.to_file(output_file='output.html')
