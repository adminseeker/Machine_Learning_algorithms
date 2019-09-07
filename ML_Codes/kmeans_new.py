import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Mall_Customers.csv")
x=dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
        kmeans=KMeans(n_clusters=i, init='k-means++',max_iter=300,n_init=10,random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

kmeans=KMeans(n_clusters=5, init='k-means++',max_iter=300,n_init=10,random_state=0)
Y_kmeans=kmeans.fit_predict(x)

plt.scatter(x[Y_kmeans==0,0],x[Y_kmeans==0,1], s=100, c='red',label='Careful')
plt.scatter(x[Y_kmeans==1,0],x[Y_kmeans==1,1], s=100, c='blue',label='Standard')
plt.scatter(x[Y_kmeans==2,0],x[Y_kmeans==2,1], s=100, c='green',label='Targets')
plt.scatter(x[Y_kmeans==3,0],x[Y_kmeans==3,1], s=100, c='cyan',label='Careless')
plt.scatter(x[Y_kmeans==4,0],x[Y_kmeans==4,1], s=100, c='magenta',label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="yellow",label='centroid')
plt.title('Clusters of clients')
plt.xlabel('Anual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()