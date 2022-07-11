import pandas as pd
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/Iris.csv")
print(df.head(5))
df.drop('Id',axis=1,inplace=True) # Id not needed
print(df.info())
print(df.isnull().sum()) #isna also same

X = df.drop('Species',axis=1) # returns all matrix except 'Species' column
y = df['Species']  # returns only 'Species' column
# Splitting values
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.70,random_state=1)
# Model Building
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train,y_train)
# Prediction
y_pred = model.predict(X_test)
print(y_pred)
#report
print(classification_report(y_test,y_pred))
#confusionmatrix
mat = confusion_matrix(y_test,y_pred)
print(mat)
ConfusionMatrixDisplay(mat).plot()
plt.show()
# Graph draw
plt.figure(figsize=(12,7))
# tree plot fn
plot_tree(model,feature_names=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],
          class_names=['Iris-setosa','Iris-versicolor','Iris-setosa'],
          filled=True)
plt.show()