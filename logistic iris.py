# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:52:25 2018

@author: hp
"""

import numpy as np
import pandas as pd

df1=pd.read_csv("Iris.csv")
features=df1.iloc[:,1:5].values
labels=df1.iloc[:,5:6].values


#df2=pd.DataFrame(labels)

#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#le1=LabelEncoder()
#labels[:,0]=le1.fit_transform(labels[:,0])

#onehotencoder=OneHotEncoder(categorical_features=[0])
#labels=onehotencoder.fit_transform(labels).toarray()

labels=le1.fit_transform(labels)

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)


from sklearn.linear_model import LogisticRegression
reg1=LogisticRegression()
reg1.fit(features_train,labels_train)

pred=reg1.predict(features_test)

Score=reg1.score(features_train,labels_train)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,pred)


