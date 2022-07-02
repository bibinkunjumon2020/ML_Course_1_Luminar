
# Classifier Algorithm

from sklearn.neighbors import KNeighborsClassifier as KNC
x1=[7,7,3,1] # Acid Durability : Feature 1
x2=[7,4,4,4] # Strength : Feature 2
x3=[6,3,5,7]
train_output=['bad','bad','good','good'] #Classification based on features
train_input=list(zip(x1,x2,x3))
test_input=[[3,6,4]]
knn=KNC(3)
knn.fit(train_input,train_output)
test_prediction=knn.predict(test_input)
print(test_prediction)

'''
We can give input as x1,x2,x3,......... and corresponding training o/p 
Can predict outcome based on these

'''
