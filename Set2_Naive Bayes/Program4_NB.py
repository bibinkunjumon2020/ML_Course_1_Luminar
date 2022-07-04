
# ------ Naive Bayes - GaussianNB --- diabetic data process
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from collections import Counter
df=pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/diabetes.csv")
# Check for null values - if there remove it or fill it
print(df.isna().sum())
print(df.head())

#---take data as features & labels

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

# DATA SET OUTPUT CHECKING - balanced or imbalanced.IF binary output has similar o/p count,then Balanced,otherwise imbalanced
# depends on counter output we can decide which algorithm is better
print(Counter(y))

#------ create training and test sets

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=42)

#----Scale

scalar=StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)


# Fitting a model

model=GaussianNB()
model.fit(X_train,y_train)


#predict

y_predict=model.predict(X_test)
#print(y_predict)

#----confusion matrix

mat=confusion_matrix(y_test, y_predict)
print(mat)

#---accuracy

accuracy=accuracy_score(y_test, y_predict)
print(accuracy)

#----plot
# Always confusion matrix labels are different categories in output.Here 0 or 1 ie:Diabetic or Non-Diabetic
md=ConfusionMatrixDisplay(mat,display_labels=['No Diabetic','Diabetic']) # Keep in mind 0 comes first so Non-Diabetic

# -- plotting the data
md.plot()
plt.show()
