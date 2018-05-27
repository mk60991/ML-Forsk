# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:42:39 2018

@author: hp
"""

import numpy as np
import pandas as pd
df1=pd.read_csv("Data.csv")

for i in ["Country","Purchased"]:
     df1[i] = df1[i].fillna(df1[i].mode()[0])
     
for i in ["Age","Salary"]:
    df1[i]=df1[i].fillna(df1[i].mean())

features=df1.iloc[:,0:3].values
labels=df1.iloc[:,3:4].values

#from sklearn.preprocessing import Imputer
#imp1=Imputer(missing_values="NaN",strategy="mean",axis=0)
#imp1=imp1.fit(features[:,1:3])
#features[:,1:3]=imp1.transform(features[:,1:3])

df2=pd.DataFrame(features)
df3=pd.DataFrame(labels)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le1=LabelEncoder()
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
print(features_train)
print(features_test)

from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
reg1.fit(features_train,labels_train)

pred=reg1.predict(features_test)

Score=reg1.score(features_train,labels_train)

