
# Simple Linear Regression - Only one dep.var & only one indep. var
# Salary :dependent variable, Yr of Experience - independent variable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/Salary_Data.csv")
print(df.head())
#----------------------
# Provides some information regarding the columns in the data
print("******Info*****")
print(df.info())
print("*****count*****")
print(df.count())

print("*****isna***")
print(df.isna().sum())
print("******describe*****")
print(df.describe())
# ------------------------------------

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=1)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
# -- plotting
plt.scatter(X_train,y_train,color='red')
plt.plot(X_test,y_pred,color='blue')
plt.xlabel('Years Exp')
plt.ylabel('Salary')
plt.title('Salary*Experience')
plt.show()

print("Coefficient/Slope : ",model.coef_)
print("y intercept :",model.intercept_)