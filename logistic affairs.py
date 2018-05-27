# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:36:03 2018

@author: hp
"""

import numpy as np
import pandas as pd

df1=pd.read_csv("affairs.csv")
features=df1.iloc[:,0:8].values
labels=df1.iloc[:,8:9].values


from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
reg1= LogisticRegression(random_state=0)
reg1.fit(features_train,labels_train)

pred=reg1.predict(features_test)

Score=reg1.score(features_train,labels_train)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,pred)


