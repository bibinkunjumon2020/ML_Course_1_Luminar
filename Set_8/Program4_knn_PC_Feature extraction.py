# implementing principle component Feature extraction for better result
# Multi Classifier Program - More than 2 groups in final Label


# Below we import full package
import pandas as pd
import matplotlib.pyplot as plt
# Below we import only specific Classes from those package
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS
from sklearn.neighbors import KNeighborsClassifier as KNC
from collections import Counter
from sklearn.decomposition import PCA
#----- Read file create dataframe
df=pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/Iris.csv")

# Check for null values - if there remove it or fill it
print(df.isna().sum())
print(df.head())

#----- Cleaning DF by dropping unnecessary columns

df.drop(['Id'],axis=1,inplace=True) # Axis must ,otherwise id not found error
print(df.tail())

#----- Seperating Input and Output Values

X=df.iloc[:,:-1]

y= df.iloc[:, -1]


# DATA SET OUTPUT CHECKING - balanced or imbalanced.IF binary output has similar o/p count,then Balanced,otherwise imbalanced
# depends on counter output we can decide which algorithm is better
print(Counter(y))

X_train, X_test, y_train, y_test=tts(X, y, train_size=0.70,random_state=1) # Keep order in mind if variable first input then tts also input first

# Scaling

scalar=SS()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# ------- Feature extraction PC
pca = PCA(n_components=3)
# I can specify component count here depends on feature concentration
# Here I adopt 3 components only but getting same 97 percent accuracy. SO in huge feature list project we can adopt this
# method to get rid of unwanted features.
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("Feature extraction PC : ",pca.explained_variance_ratio_)
#classification :KNN ie:predicting

knn = KNC(n_neighbors=7)  #model created
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
#print(y_pred)



#----- C.Matrix and accuracy

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# --------- Display Matrix using Matplot


labels = ['Versicolor','Setosa','Virginica'] #on X and y axis op labels
md = ConfusionMatrixDisplay(matrix,display_labels=labels)

md.plot()
#plt.show() #It pops up the window to show graph