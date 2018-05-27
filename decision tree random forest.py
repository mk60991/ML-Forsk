# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:51:39 2018

@author: hp
"""

import pandas as pd
import numpy as np

df1=pd.read_csv("PastHires.csv")
features=df1.iloc[:,0:6].values
labels=df1.iloc[:,6].values

df2=pd.DataFrame(features)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
for i in [1,3,4,5]:
    x[:,i]=labelencoder.fit_transform(x[:,i])
y=labelencoder.fit_transform(y)

#le1=LabelEncoder()
#features[:,1]=le1.fit_transform(features[:,1])

#features[:,3]=le1.fit_transform(features[:,3])
#features[:,4]=le1.fit_transform(features[:,4])


labels=le1.fit_transform(labels)

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)


from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(features,labels)

pred=regressor.predict(features)


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(features,labels)

emp=np.array([10,1,4,0,1,0]).reshape(1,-1)
un_emp=np.array([10,0,4,1,0,0]).reshape(1,-1)

pred_emp=regressor.predict(emp)
pred_un_emp=regressor.predict(un_emp)

