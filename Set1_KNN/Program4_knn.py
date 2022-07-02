# Multi Classifier Program - More than 2 groups in final Label

# Below we import full package
import pandas as pd
import matplotlib.pyplot as plt
# Below we import only specific Classes from those package
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS
from sklearn.neighbors import KNeighborsClassifier as KNC


#----- Read file create dataframe
df=pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/Iris.csv")
print(df.head())

#----- Cleaning DF by dropping unnecessary columns

df.drop(['Id'],axis=1,inplace=True) # Axis must ,otherwise id not found error
print(df.tail())

#----- Seperating Input and Output Values

X=df.iloc[:,:-1].values
#print(X)
Y=df.iloc[:,-1].values
#print(Y)

X_train,X_test,Y_train,Y_test=tts(X,Y,train_size=0.70) # Keep order in mind if variable first input then tts also input first

# Scaling

scalar=SS()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

#classification :KNN ie:predicting

knn=KNC(n_neighbors=7)  #model created
knn.fit(X_train,Y_train)
Y_predict=knn.predict(X_test)
#print(Y_predict)

# Random check

print("Random Prediction 1 : ",knn.predict([[4.9,3.0,1.4,0.2]])) # Here I get WRONG prediction

print("Random Prediction 2: ",knn.predict([[5.9,3,5.1,1.8]])) #Here True Prediction

#----- C.Matrix and accuracy

matrix=confusion_matrix(Y_test,Y_predict)
print(matrix)
accuracy=accuracy_score(Y_test,Y_predict)
print(accuracy)

# --------- Display Matrix using Matplot


labels=['Versicolor','Setosa','Virginica'] #on X and Y axis op labels
md=ConfusionMatrixDisplay(matrix,display_labels=labels)

md.plot()
plt.show() #It pops up the window to show graph