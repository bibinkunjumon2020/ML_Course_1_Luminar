
# Gaussian Naive Bayers algorithm - GaussianNB : Iris Data

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris # I used this data instead of importing file and making dataframe
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay

# ---- Data is imported from dataset library

irisData=load_iris()
print("Column Headings(Features) : ",irisData.feature_names)
print('Outputs(label target) : ',irisData.target_names)

#Data loading

X=irisData.data  #features set
y=irisData.target #target set
print(X,y)

# splitting data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)

# binding with GaussianNB to model
model=GaussianNB()
model.fit(X_train,y_train)

#predicting
y_predict = model.predict(X_test)
# Printing the prediction actual values(bcs here 0 and 1 in target set)
print('Prediction set: ',irisData.target_names[y_predict])

y_random = model.predict([[5.1,3.5,1.4,0.2]])  # Success.....!
print('Random prediction : ',y_random,"Value is : ",irisData.target_names[y_random])
y_random2 = model.predict([[5.9,3,5.1,1.8]])   # Success......!
print('Random prediction 2 : ',y_random2,"Value is : ",y_random2)

# --- accuracy score
accuracy=accuracy_score(y_test,y_predict)
print('Accuracy : =',accuracy)  # Accuracy = 1 i got

#----Confusion Matrix

cm=confusion_matrix(y_test,y_predict)
print("Confusion Matrix : \n",cm)

# --classification report
cmr=classification_report(y_test,y_predict)
print("Classification Report : \n",cmr)

#---Display ,plot confusion matrix

md=ConfusionMatrixDisplay(cm,display_labels=irisData.target_names)
md.plot()
plt.show()