
# Lung Cancer data - set processing using KNN,NB,SVM algorithms
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from matplotlib import pyplot as plt
from collections import Counter
# Dataframe

df=pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/lung_cancer_examples.csv")

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
df.isna().sum() # No Nulls
df.head() # Name	Surname	Age	Smokes	AreaQ	Alkhol	Result - Here unwanted items are there

# Data cleaning-dropping unwanted columns from prediction process

df.drop(['Name','Surname'],axis=1,inplace=True) #Axis=1 is must otherwise 'id' not found error
df.head()

# Data Seperate - X y

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# DATA SET OUTPUT CHECKING - balanced or imbalanced.IF binary output has similar o/p count,then Balanced,otherwise imbalanced
# depends on counter output we can decide which algorithm is better
print(Counter(y))

#  Scaling the whole input set -Here i have done full set instead of X_train
scalar = MinMaxScaler()
# Here I used MinMaxScalar instead of StandardScalar because output of SS is -ve numbers too,
# that not supported by MultinomialNB algoritham.Error throwing up
scalar.fit(X)
X = scalar.transform(X)

# Splitting data

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.75,random_state = 1)
# --------------------------------------------------------------------------
print('*'*20,"KNN Model",'*'*20)
# Model Build

model = KNeighborsClassifier(3)
model.fit(X_train,y_train)

# Predict

y_pred = model.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of KNN Model : ",accuracy)

# Confusion Matrix

mat = confusion_matrix(y_test,y_pred)
print("Matrix is    :\n",mat)

# Matrix plot

md = ConfusionMatrixDisplay(mat,display_labels=['No Cancer','Yes Cancer'])
md.plot()
plt.show()
# ----------------------------------------------------------------------------
print('*'*20,"Naive Bayes Model - MultinomialNB",'*'*20)

# Model Build

model_M = MultinomialNB()
model_M.fit(X_train,y_train)

# Predict

y_pred = model_M.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of KNN Model : ",accuracy)

# Confusion Matrix

mat_M = confusion_matrix(y_test,y_pred)
print("Matrix is    :\n",mat_M)

# Matrix plot

md_M = ConfusionMatrixDisplay(mat_M,display_labels=['No Cancer','Yes Cancer'])
md_M.plot()
plt.show()

# ----------------------------------------------------------------------------
print('*'*20,"Naive Bayes Model - GaussianNB",'*'*20)

# Model Build

model_G = GaussianNB()
model_G.fit(X_train,y_train)

# Predict

y_pred = model_G.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of KNN Model : ",accuracy)

# Confusion Matrix

mat_G = confusion_matrix(y_test,y_pred)
print("Matrix is    :\n",mat_G)

# Matrix plot

md_G = ConfusionMatrixDisplay(mat_G,display_labels=['No Cancer','Yes Cancer'])
md_G.plot()
plt.show()

# ----------------------------------------------------------------------------
print('*'*20,"Naive Bayes Model - BernoulliNB",'*'*20)

# Model Build

model_B = BernoulliNB()
model_B.fit(X_train,y_train)

# Predict

y_pred = model_B.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of KNN Model : ",accuracy)

# Confusion Matrix

mat_B = confusion_matrix(y_test,y_pred)
print("Matrix is    :\n",mat_B)

# Matrix plot

md_B = ConfusionMatrixDisplay(mat_B,display_labels=['No Cancer','Yes Cancer'])
md_B.plot()
plt.show()

#----------------------------------------------------------------------------
print('*'*20,"Support Vector Machines-SVC",'*'*20)

#Model Build

model_C = SVC()
model_C.fit(X_train,y_train)

#Predict

y_pred = model_C.predict(X_test)

#Accuracy

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of KNN Model : ",accuracy)

#Confusion Matrix

mat_C = confusion_matrix(y_test,y_pred)
print("Matrix is    :\n",mat_C)

#Matrix plot

md_C = ConfusionMatrixDisplay(mat_C,display_labels=['No Cancer','Yes Cancer'])
md_C.plot()
plt.show()