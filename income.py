# -*- coding: utf-8 -*-
"""
Created on Sun May 20 15:47:26 2018

@author: hp
"""

import numpy as np
import pandas as pd
df=pd.read_csv("Income_Data.csv")
features=df.iloc[:,0:1].values
labels=df.iloc[:,1:2].values


from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.3,random_state=0)


from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
reg1.fit(features_train,labels_train)

pred=reg1.predict(features_test)

Score=reg1.score(features_train,labels_train)

import matplotlib.pyplot as plt
plt.scatter(features_train,labels_train,color="red")
plt.plot(features_train,reg1.predict(features_train),color='blue')
plt.title("income vs ml")
plt.xlabel("ml")
plt.ylabel("income")
plt.show()

import matplotlib.pyplot as plt
plt.scatter(features_test,labels_test,color="red")
plt.plot(features_test,reg1.predict(features_test),color='blue')
plt.title("income vs ml")
plt.xlabel("ml")
plt.ylabel("income")
plt.show()

import matplotlib.pyplot as plt
plt.plot(features,labels,color="red")
#plt.plot(features_train,reg1.predict(features_train),color='blue')
plt.title("income vs ml")
plt.xlabel("ml")
plt.ylabel("income")
plt.show()