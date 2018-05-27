# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:47:54 2018

@author: hp
"""

import numpy as np
import pandas as pd
df=pd.read_json("http://www.openedx.forsk.in/c4x/Forsk_Labs/FL102/asset/spam_ham.json")
x=df.head()


#drop unwanted column 
features=df.iloc[:,4:5].values
labels=df.iloc[:,3:4].values
df1=pd.DataFrame(features)
df2=pd.DataFrame(labels)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le1=LabelEncoder()
features[:,0]=le1.fit_transform(features[:,0])

labels=le1.fit_transform(labels)

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
reg1.fit(features_train,labels_train)

pred=reg1.predict(features_test)

Score=reg1.score(features_train,labels_train)

