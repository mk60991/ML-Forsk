# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 09:12:52 2018

@author: hp
"""

import pandas as pd
import numpy as np

df1=pd.read_csv("iq_size.csv")
features=df1.iloc[:,1:4].values
labels=df1.iloc[:,0].values

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
reg1.fit(features_train,labels_train)

lst1=np.array([90,70,150])
lst1.reshape(1,3)
pred=reg1.predict([lst1])

Score=reg1.score(features_train,labels_train)

#pred=reg1.predict(features_test)

#Score=reg1.score(features_train,labels_train)

