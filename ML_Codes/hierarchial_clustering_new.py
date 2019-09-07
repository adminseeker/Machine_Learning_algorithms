import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Mall_Customers.csv")
x=dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
Y_hc=hc.fit_predict(x)

plt.scatter(x[Y_hc==0,0],x[Y_hc==0,1], s=100, c='red',label='Careful')
plt.scatter(x[Y_hc==1,0],x[Y_hc==1,1], s=100, c='blue',label='Standard')
plt.scatter(x[Y_hc==2,0],x[Y_hc==2,1], s=100, c='green',label='Targets')
plt.scatter(x[Y_hc==3,0],x[Y_hc==3,1], s=100, c='cyan',label='Careless')
plt.scatter(x[Y_hc==4,0],x[Y_hc==4,1], s=100, c='magenta',label='Sensible')
plt.title('Clusters of clients')
plt.xlabel('Anual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
