import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('data.csv')
mass = list(df.iloc[:,0])
radius = list(df.iloc[:,1])
gravity = list(df.iloc[:,2])

X = df.iloc[:,[0,1]].values.tolist()
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=9)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig = plt.figure(figsize=(10,5))
sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WSS")
# TODO fig.show()  

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=9)
cluster_index = kmeans.fit_predict(X)

# [[  1.87580321   7.02204835]
# [193.         867.33333333]
# [ 31.92571429 319.71428571]]
clus_1 = []
clus_2 = []
clus_3 = []

for index, data in enumerate(cluster_index):
    if data == 0:
        clus_1.append(index)
    elif data == 1:
        clus_2.append(index)
    elif data == 2:
        clus_3.append(index)

fig = plt.figure(figsize=(10,5))
sns.scatterplot(x=[X[i][0] for i in clus_1], y=[X[i][1] for i in clus_1], color='yellow', label='Cluster_1')
sns.scatterplot(x=[X[i][0] for i in clus_2], y=[X[i][1] for i in clus_2], color='blue', label='Cluster_2')
sns.scatterplot(x=[X[i][0] for i in clus_3], y=[X[i][1] for i in clus_3], color='red', label='Cluster_3')
sns.scatterplot(x = kmeans.cluster_centers_[:,0], y = kmeans.cluster_centers_[:,1], color='black', lable='Centroid', s=100, markers=',')
plt.grid(False)
plt.xlabel("Mass")
plt.ylabel("Radius")
plt.title("Kmeans-Clustering")
plt.legend()
fig.show()