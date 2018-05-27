# -*- coding: utf-8 -*-
"""
Created on Mon May 21 08:57:45 2018

@author: hp
"""

import numpy as np
import pandas as pd

df1=pd.read_csv("codebeautify.csv")


for i in ["v1","FIELD3","FIELD4","FIELD5"]:
     df1[i] = df1[i].fillna(df1[i].mode()[0])
     
labels=df1.iloc[:,0:1].values
features=df1.iloc[:,1:5].values

dfx=pd.DataFrame(features)
dfy=pd.DataFrame(labels)


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le1=LabelEncoder()
features[:,1]=le1.fit_transform(features[:,1])
features[:,2]=le1.fit_transform(features[:,2])
features[:,3]=le1.fit_transform(features[:,3])
features[:,0]=le1.fit_transform(features[:,0])

onehotencoder=OneHotEncoder(categorical_features=[0])
features=onehotencoder.fit_transform(features).toarray()

features=features[:,1:]

labels=le1.fit_transform(labels)


from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
features_train=sc1.fit_transform(features_train)
features_test=sc1.transform(features_test)


from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
reg1.fit(features_train,labels_train)

pred=reg1.predict(features_test)

Score=reg1.score(features_train,labels_train)

