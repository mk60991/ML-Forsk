# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 09:58:24 2018

@author: hp
"""

import numpy as np
import pandas as pd

df1=pd.read_csv("Bahubali2_vs_Dangal.csv")
features=df1.iloc[:,:1].values
labels1=df1.iloc[:,1:2].values
labels2=df1.iloc[:,2:3].values


#from sklearn.model_selection import train_test_split
#features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression

reg1=LinearRegression()
reg1.fit(features,labels1)

pred1=reg1.predict(10)

Score1=reg1.score(features,labels1)


import matplotlib.pyplot as plt
plt.scatter(features,labels1,color="red")
plt.plot(features,reg1.predict(features),color="blue")
plt.title("days vs bahubali2")
plt.xlabel("days")
plt.ylabel("bahubali")
plt.show()



reg2=LinearRegression()
reg2.fit(features,labels2)

pred2=reg2.predict(10)

Score2=reg2.score(features,labels2)


plt.scatter(features,labels2,color="red")
plt.plot(features,reg2.predict(features),color="blue")
plt.title("days vs bahubali2")
plt.xlabel("days")
plt.ylabel("dangal")
plt.show()