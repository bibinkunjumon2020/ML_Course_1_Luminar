# Regression problem using Tree graph

"""
1.Classifier gives an output in which group/Yes,NO/1,0
2.Regressor gives a value out put like 23.5
3.To analyse data in Regressor - we use graph,root mean square,etc
4.To analuse Classifier we use precision,accuracy,confusion matrix,accuracy score etc

"""



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv("/Source/advertising.csv")
print(df.head(5))
print(df.info())
# Separating X & y
X = df.drop('Sales',axis=1)
y= df['Sales']
# Split train and test datas
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.70,random_state=1)
# Build model
model = DecisionTreeRegressor() # Here we get values as output
model.fit(X_train,y_train)
# Predict
y_pred = model.predict(X_test)
print(y_pred)
# graph
plot_tree(model,feature_names=['TV','Radio','Newspaper'],filled=True)
plt.show()

print("Mean Absolute Error")
print(mean_absolute_error(y_test,y_pred))
print("Mean Squared Error")
print(mean_squared_error(y_test,y_pred))
print("R2score")
print(r2_score(y_test,y_pred))