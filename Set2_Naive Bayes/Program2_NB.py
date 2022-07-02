
# Solving Iris data set using Naive Bayer's - MultinomialNB

from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# Loading Data  ---- Data is imported from dataset library
irisData=load_iris()

print("Input Features  :",irisData.feature_names)

print("Target Labels : ",irisData.target_names)

X=irisData.data
Y=irisData.target
print(X)
# -----splitting data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)

# ------Building and fitting model
model = MultinomialNB()
model.fit(X_train,Y_train)
# predicting
Y_predict = model.predict(X_test)
print(Y_predict,'\n',Y_test)

# ----checking accuracy

mtr=confusion_matrix(Y_test,Y_predict)
print(mtr)
accuracy=accuracy_score(Y_test,Y_predict)
print(accuracy)

md=ConfusionMatrixDisplay(mtr,display_labels=irisData.target_names)
md.plot()
plt.show()