# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:53:24 2018

@author: hp
"""

import numpy as np
import pandas as pd

df1=pd.read_csv("deliveryfleet.csv")
features=df1.iloc[:,1:3].values

# wcss used to determine number of cluster should be formed
#which form maximunm variation
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("the elbow method")
plt.xlabel("numbers of clusters")
plt.ylabel("wcss")
plt.show()


kmeans=KMeans(n_clusters=2,init="k-means++",random_state=0)
y_means=kmeans.fit_predict(features)

plt.scatter(features[y_means==0,0],features[y_means==0,1],s=100,c="red",label="rurals")
plt.scatter(features[y_means==1,0],features[y_means==1,1],s=100,c="blue",label="urbans")
plt.scatter(features[y_means==2,0],features[y_means==2,1],s=100,c="brown",label="rurals")
plt.scatter(features[y_means==3,0],features[y_means==3,1],s=100,c="black",label="urbans")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c="yellow",label="centroids")
plt.title("clusters of customers")
plt.xlabel("distance features")
plt.ylabel("speeding faetures")
plt.legend()
plt.show()


kmeans=KMeans(n_clusters=4,init="k-means++",random_state=0)
y_means=kmeans.fit_predict(features)

plt.scatter(features[y_means==0,0],features[y_means==0,1],s=100,c="red",label="rurals")
plt.scatter(features[y_means==1,0],features[y_means==1,1],s=100,c="blue",label="urbans")
plt.scatter(features[y_means==2,0],features[y_means==2,1],s=100,c="brown",label="rurals")
plt.scatter(features[y_means==3,0],features[y_means==3,1],s=100,c="black",label="urbans")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c="yellow",label="centroids")
plt.title("clusters of customers")
plt.xlabel("distance features")
plt.ylabel("speeding faetures")
plt.legend()
plt.show()