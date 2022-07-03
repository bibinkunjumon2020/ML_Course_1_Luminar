
# Support Vector Classification from Support Vector Machine
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,confusion_matrix
from sklearn.svm import SVC
from matplotlib import pyplot as plt



df = pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/diabetes.csv")

# data separating

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# data splitting

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=1)

# data scaling

scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

# Machine model

model = SVC()
model.fit(X_train,y_train)

# Prediction

y_predict = model.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test,y_predict)
print(f"Accuracy of the Model SVC = {accuracy}")

# Confusion Matrix

mat = confusion_matrix(y_test,y_predict)
print(mat)

# Display Confusion Matrix

dis=ConfusionMatrixDisplay(mat,display_labels=['Non Diabetic','Diabetic'])
dis.plot()
plt.show()
