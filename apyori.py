# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 13:05:20 2018

@author: hp
"""

import pandas as pd
from apyori import apriori
df1=pd.read_csv("Market_Basket_Optimisation.csv",header=None)
transactions=[]
for i in range(7501):
    transaction.append([str(data.values[i,j]) for j in range(20)])


rules=apriori(transaction,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

results=list(rules)