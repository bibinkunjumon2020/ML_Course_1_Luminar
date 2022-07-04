
# Classifier Algorithm

from sklearn.neighbors import KNeighborsClassifier as KNC
x1=[7,7,3,1] # Acid Durability : Feature 1
x2=[7,4,4,4] # Strength : Feature 2
y_train=['bad', 'bad', 'good', 'good'] #Classification based on features
X_train=list(zip(x1, x2))
X_test=[[3, 6]]
knn=KNC(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
print(y_pred)

