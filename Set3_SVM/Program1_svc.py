
# Support Vector Classification from Support Vector Machine
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,confusion_matrix
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from collections import Counter


df = pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/diabetes.csv")

#----------------------
# Provides some information regarding the columns in the data
print("******Info*****")
print(df.info())
print("*****count*****")
print(df.count())
print("*****isna***")
print(df.isna().sum())
print("******describe*****")
print(df.describe())

# ------------------------------------
# Check for null values - if there remove it or fill it
print(df.isna().sum())
# data separating

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# DATA SET OUTPUT CHECKING - balanced or imbalanced.IF binary output has similar o/p count,then Balanced,otherwise imbalanced
# depends on counter output we can decide which algorithm is better
print(Counter(y))
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
