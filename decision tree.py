# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:42:30 2018

@author: hp
"""

import pandas as pd
import numpy as np

df1=pd.read_csv("Position_Salaries.csv")
features=df1.iloc[:,1:2].values
labels=df1.iloc[:,2].values

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)


from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(features,labels)

pred=regressor.predict(11.5)