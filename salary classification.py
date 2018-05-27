# -*- coding: utf-8 -*-
"""
Created on Sun May 20 16:08:19 2018

@author: hp
"""
import numpy as np
import pandas as pd
df=pd.read_csv("Salary_Classification.csv")

features=df.iloc[:,0:4].values
labels=df.iloc[:,-1:].values

df1=pd.DataFrame(features)
df2=pd.DataFrame(labels)


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le1=LabelEncoder()
features[:,0]=le1.fit_transform(features[:,0])


onehotencoder=OneHotEncoder(categorical_features=[0])
features=onehotencoder.fit_transform(features).toarray()

features=features[:,1:]

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





