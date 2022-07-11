import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree


df = pd.read_csv("/Source/lung_cancer_examples.csv")
print(df.head(5))
print(df.info())
# drop unnecessary features
df.drop(['Name','Surname'],axis=1,inplace=True)
print(df.info())
# Separate features & Labels
X = df.drop('Result',axis=1)
y = df['Result']
# train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.75,random_state=1)
# model fitting
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train,y_train)
#predict
y_pred = model.predict(X_test)
print(y_pred)
#graph
plt.figure(figsize=(8,9))
plot_tree(model,feature_names=['Age','Smokes','AreaQ','Alkhol'],filled=True)
plt.show()
