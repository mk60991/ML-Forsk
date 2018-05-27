# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:34:59 2018

@author: hp
"""

import numpy as np
import pandas as pd

df1=pd.read_csv("Social_Network_Ads.csv")
features=df1.iloc[:,[2,3]].values
labels=df1.iloc[:,4].values

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
features_train=sc1.fit_transform(features_train)
features_test=sc1.transform(features_test)


from sklearn.linear_model import LogisticRegression
reg1=LogisticRegression()
reg1.fit(features_train,labels_train)

pred=reg1.predict(features_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,pred)