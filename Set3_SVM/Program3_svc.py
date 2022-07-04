
# Heart disease dataset using SVC under SVM

import pandas as pd
from sklearn.preprocessing import minmax_scale,robust_scale
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt
from collections import Counter


df = pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/heart (1).csv")
#print(df.head())

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
# DATA SET OUTPUT CHECKING - balanced or imbalanced.IF binary output has similar o/p count,then Balanced,otherwise imbalanced
# depends on counter output we can decide which algorithm is better
print(Counter(y))

X = robust_scale(X)

#X = minmax_scale   : This also correct we can use this function directly.No need of fit and transform
#scalar=MinMaxScaler - Class object
# or scalar = StandardScaler - class object it has no function
#scalar.fit(X)
#X = scalar.transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.75,random_state=1)

model = SVC()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of Model : ",accuracy)

mat = confusion_matrix(y_test,y_pred)
print("Confusion Matrix :\n",mat)

report = classification_report(y_test,y_pred)
print("Report : \n",report)

mat_dis = ConfusionMatrixDisplay(mat,display_labels=['No','Yes'])
mat_dis.plot()
plt.show()
