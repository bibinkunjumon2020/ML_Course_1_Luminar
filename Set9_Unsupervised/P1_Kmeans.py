# Unsupervised algorithm ..so source has no Label field. Algorithm find itself similarities and make into groups called
# clusters. So final putput has a Label or outcome

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/Mall_Customers.csv")
print(df.head(5))

X = df.drop(['CustomerID','Genre','Age'],axis=1)
print(X.head(5))

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init="k-means++",random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5,init="k-means++",random_state=42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

X['Cluster'] = y_kmeans
print(X.head(5))