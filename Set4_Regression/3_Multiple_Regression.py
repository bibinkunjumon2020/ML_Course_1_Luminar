
# Multiple Linear Regression - Multiple features(input values)
# Over fitting and Under fitting
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
# -- import file and created dataset frame
df = pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/advertising.csv")
#print(df.head())

# --- separate features & labels
X = df.drop(['Sales'],axis=1)
y = df['Sales']

# --- Scaling (no need of fiting and transform in this scaling,directly fn called)
X = minmax_scale(X,axis=0)

# --- splitting values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

# Model Algo selection

model = LinearRegression()
model.fit(X_train,y_train)

# -- prediction
y_pred = model.predict(X_test)

print("Intercept - ",model.intercept_)
print("Slope - (a+xb) x = \n",list(zip(model.coef_,X)))
#print("Prediction : \n",y_pred)

df = pd.DataFrame({"Actual Value":y_test,"Predicted Value":y_pred})
print(df.head())

# Performance Evaluation
print("Mean Absolute Error : ",mean_absolute_error(y_test,y_pred))
print("Mean Squared Error : ",mean_squared_error(y_test,y_pred))
print("Route Mean Squared Error : ",np.sqrt(mean_absolute_error(y_test,y_pred)))