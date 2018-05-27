# -*- coding: utf-8 -*-
"""
Created on Sun May 20 10:13:40 2018

@author: hp
"""

import numpy as np
import pandas as pd
df1=pd.read_csv('Cars.csv')
for i in ["Mileage"]:
    df1[i]=df1[i].fillna(df1[i].median())
print(df1)

df2=df1.iloc[:,0:3]

df2.to_csv('cars_imputed.csv')

pd.read_csv('cars_imputed.csv')