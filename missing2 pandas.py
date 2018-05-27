# -*- coding: utf-8 -*-
"""
Created on Sun May 20 11:05:15 2018

@author: hp
"""
import numpy as np
import pandas as pd

df=pd.read_csv("training_titanic.csv")
labels=df.iloc[:,1:2].values
features=df.iloc[:,2:12].values
df1=pd.DataFrame(features)
df2=pd.DataFrame(labels)

for i in ["Cabin","Embarked"]:
     df[i] = df[i].fillna(df[i].mode()[0])
     
for i in ["Age"]:
    df[i]=df[i].fillna(df[i].mean())
    
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le1=LabelEncoder()
features[:,1]=le1.fit_transform(features[:,1])
features[:,2]=le1.fit_transform(features[:,2])
features[:,6]=le1.fit_transform(features[:,6])
features[:,8]=le1.fit_transform(features[:,8])
features[:,9]=le1.fit_transform(features[:,9])


from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
reg1.fit(features_train,labels_train)

pred=reg1.predict(features_test)

Score=reg1.score(features_train,labels_train)

