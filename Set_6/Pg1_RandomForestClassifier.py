import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,RocCurveDisplay,roc_auc_score

df = pd.read_csv("/Users/bibinkunjumon/PycharmProjects/ML_Course_1_Luminar/Source/bill_authentication.csv")
print(df.head(5))

X = df.drop('Class',axis=1)
y = df['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.75,random_state=1)

model = RandomForestClassifier(n_estimators=20)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(y_pred)

print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print("Class Report : \n",classification_report(y_test,y_pred))
print("Accuracy Score = ",accuracy_score(y_test,y_pred))

print("roc_auc_score = ",roc_auc_score(y_test,y_pred))

#plot_roc_curve(model,X_test,y_test)  - deprecated

RocCurveDisplay.from_estimator(model,X_test,y_test)
#RocCurveDisplay.from_predictions(y_test,y_pred)  - It can be also used
plt.show()

