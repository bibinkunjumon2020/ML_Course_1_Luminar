# Multi Classifier Program - More than 2 groups in final Label
# GridSearch used to improve accuracy

# Below we import full package
import pandas as pd
import matplotlib.pyplot as plt
# Below we import only specific Classes from those package
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler as SS
from sklearn.neighbors import KNeighborsClassifier as KNC
from collections import Counter

#----- Read file create dataframe
df=pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/Iris.csv")

# Check for null values - if there remove it or fill it
print(df.isna().sum())
print(df.head())

#----- Cleaning DF by dropping unnecessary columns

df.drop(['Id'],axis=1,inplace=True) # Axis must ,otherwise id not found error
print(df.tail())

#----- Seperating Input and Output Values

X=df.iloc[:,:-1].values

y= df.iloc[:, -1].values


# DATA SET OUTPUT CHECKING - balanced or imbalanced.IF binary output has similar o/p count,then Balanced,otherwise imbalanced
# depends on counter output we can decide which algorithm is better
print(Counter(y))

X_train, X_test, y_train, y_test=tts(X, y, train_size=0.70,random_state=1) # Keep order in mind if variable first input then tts also input first

# Scaling

scalar=SS()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

#classification :KNN ie:predicting

knn = KNC(n_neighbors=7,weights='uniform')  #model created
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
#print(y_pred)

# Grid Search used for finding best parameters
knn1 = KNC()
param = {'n_neighbors':[3,5,7,9],'weights':['uniform','distance']}
model = GridSearchCV(knn1,param,cv=5,scoring='accuracy')
model.fit(X_train,y_train)
print("*"*100,model.best_params_)

knn2 = KNC(n_neighbors=model.best_params_['n_neighbors'],weights=model.best_params_['weights'])
knn2.fit(X_train,y_train)
y_pred_2 = knn2.predict(X_test)

# Random check

print("Random Prediction 1 : ",knn.predict([[4.9,3.0,1.4,0.2]])) # Here I get WRONG prediction

print("Random Prediction 2: ",knn.predict([[5.9,3,5.1,1.8]])) #Here True Prediction

#----- C.Matrix and accuracy

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
accuracy2 = accuracy_score(y_test, y_pred_2)
print("Accuracy 2 =  ",accuracy2)
# --------- Display Matrix using Matplot


labels = ['Versicolor','Setosa','Virginica'] #on X and y axis op labels
md = ConfusionMatrixDisplay(matrix,display_labels=labels)

md.plot()
#plt.show() #It pops up the window to show graph