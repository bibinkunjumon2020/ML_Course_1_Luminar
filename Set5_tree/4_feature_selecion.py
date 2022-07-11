"""
Feature Selection : Choosing most correlated features for best model.Its a pre processing

"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix
import time

df = pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/mobile_train.csv")
# Here we use this train set for both training and testing
print(df.head(5))

X = df.drop('price_range',axis=1)
y = df['price_range']

best = SelectKBest(k=10) #k=10 means i need only 10 top values
best.fit(X,y)

dfscore = pd.DataFrame(best.scores_) # A data frame of scores of features
dfscore = pd.concat([dfscore,pd.DataFrame(X.columns)],axis=1) # df of columns and scores
dfscore.columns=['score','spec']  # giving column titles
print(dfscore)
print(dfscore.nlargest(10,'score')) # selecting large 10 features
# Only chose top 10 values
X_true = df[['ram','battery_power','px_width','px_height','mobile_wt','int_memory','n_cores','sc_h','sc_w','talk_time']]

print(X_true.head(5))
# splitting data
X_train,X_test,y_train,y_test = train_test_split(X_true,y,train_size=0.70,random_state=1)

start1 = time.time() # timer for measuring calculation time

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
# prediction
y_pred = model.predict(X_test)
end1 = time.time()
print(y_pred)


report = classification_report(y_test,y_pred)
print("Report = \n",report)

mat = confusion_matrix(y_test,y_pred)
print("Matrix : \n",mat)

mat_dis = ConfusionMatrixDisplay(mat)
mat_dis.plot()
#plt.show()

"""
Feature Selection method :
1.Avoid over fitting
2.Increase accuracy
3.Training time is reduced

"""
# ---- without feature selection result

X_train_old,X_test_old,y_train_old,y_test_old = train_test_split(X,y,train_size=0.70,random_state=1)

start2 = time.time()

model_old = DecisionTreeClassifier()
model_old.fit(X_train_old,y_train_old)
y_pred_old = model_old.predict(X_test_old)

end2 = time.time()

print(y_pred_old)

print("Report Old :",classification_report(y_test_old,y_pred_old))

print("Time Difference = ",(end1-start1),(end2-start2))

"""
Here in both tests no better accuracy or time i got
"""