# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:47:56 2018

@author: hp
"""

import pandas as pd
import numpy as np

df1=pd.read_csv("Social_Network_Ads.csv")
features=df1.iloc[:,0:4].values
labels=df1.iloc[:,4:5]

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le1=LabelEncoder()
features[:,1]=le1.fit_transform(features[:,1])
df2=pd.DataFrame(features)
#onehotencoder=OneHotEncoder(categorical_features=[1])
#features=onehotencoder.fit_transform(features).toarray()

#features=features[:,1:]

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
reg1.fit(features_train,labels_train)

pred=reg1.predict(features_test)

Score=reg1.score(features_train,labels_train)

