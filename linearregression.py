# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:15:35 2018

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df1=pd.read_csv("Income_Data.csv")
features=df1.iloc[:,0:1].values
labels=df1.iloc[:,1:2].values


from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
reg1.fit(features_train,labels_train)
 
 
labels_pred=reg1.predict(features_test)

Score=reg1.score(features_test,labels_test)
 

plt.scatter(features_train,labels_train,color="red")
plt.plot(features_train,reg1.predict(features_train),color="blue")
plt.title("income vs ml ecxpreieb=nsce")
plt.xlabel("ml expreincw")
plt.ylabel("income")
plt.show()

plt.scatter(features_test,labels_test,color="red")
plt.plot(features_train,reg1.predict(features_train),color="blue")
plt.title("income vs ml ecxpreieb=nsce")
plt.xlabel("ml expreincw")
plt.ylabel("income")
plt.show()