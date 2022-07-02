
# Classifier Algorithm

from sklearn.neighbors import KNeighborsClassifier as KNC
x1=[7,7,3,1] # Acid Durability : Feature 1
x2=[7,4,4,4] # Strength : Feature 2
train_output=['bad','bad','good','good'] #Classification based on features
train_input=list(zip(x1,x2))
test_input=[[3,6]]
knn=KNC(n_neighbors=3)
knn.fit(train_input,train_output)
test_output=knn.predict(test_input)
print(test_output)

