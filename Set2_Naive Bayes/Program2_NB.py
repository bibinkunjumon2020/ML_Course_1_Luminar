
# Solving Iris data set using Naive Bayer's - MultinomialNB

from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from collections import Counter
# Loading Data  ---- Data is imported from dataset library
irisData=load_iris()

print("Input Features  :",irisData.feature_names)

print("Target Labels : ",irisData.target_names)

X=irisData.data
y=irisData.target
# DATA SET OUTPUT CHECKING - balanced or imbalanced.IF binary output has similar o/p count,then Balanced,otherwise imbalanced
# depends on counter output we can decide which algorithm is better
print(Counter(y))
# -----splitting data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

# ------Building and fitting model
model = MultinomialNB()
model.fit(X_train,y_train)
# predicting
y_pred = model.predict(X_test)


# ----checking accuracy

mtr=confusion_matrix(y_test,y_pred)
print("Confusion Matrix :\n",mtr)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy = ",accuracy)

md=ConfusionMatrixDisplay(mtr,display_labels=irisData.target_names)
md.plot()
plt.show()