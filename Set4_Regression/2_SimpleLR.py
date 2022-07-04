
# Simple Linear Regression Height weight data set
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/height_weight.csv")
# Details about df
print(df.head())
print(df.info())
print(df.describe())

#X = df.iloc[:,:-1].values # with or without .values it works

# another way of picking 2D array incase of Simple Linear Regression
# bcs in SLR only 1 set feature 1 set outcome
X = df['Height'].values.reshape(-1,1)  #Here .values must for reshape into 2d array./Only one unknown index possible
# so i gave -1,another index cannot be -1, so i need column index till 1 ie:0

y = df['Weight'].values  #with or without .values it works

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.75,random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(y_pred)

# Plotting
plt.scatter(X_train,y_train,color='r',marker='*')
plt.plot(X_test,y_pred,color='b')
plt.title("Height * Weight Graph")
plt.xlabel("Height")
plt.ylabel('Weight')


print("Coefficient/Slope : ",model.coef_)
print("y intercept :",model.intercept_)

plt.show()  # always give at bottom so other codes finish execution
