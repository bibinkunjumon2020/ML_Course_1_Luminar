
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures



df = pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/book4.csv")
print(df.head())
#print(df.count())

# --Cleaning data

X = df.drop(['Pressure','Sno'],axis=1)  # Warning :LR only 1 feature column 1 Label column
y = df['Pressure']

# --- Making X as a polynomial

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
#y_poly = poly.fit_transform(y) -No need to transform y
print(X_poly)

# Linear model setting
model = LinearRegression()
model.fit(X_poly,y)
# -- Prediction

y_pred = model.predict(X_poly)
#print(y_pred)

# -- Plotting

plt.scatter(X,y,color='red')  #Here we use the same X from start otherwise error
plt.plot(X,y_pred,color='blue')
plt.title("Polynomial Regression")
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()

'''
From output graph its clear that y_pred and y actual has no relation.So no linear regression method possible here.
So we have to adopt Polynomial Regression method
c=y+mx --> d=C+Bx+Ax^2

Here with degree=4 ,output graph perfectly fits - Success
'''
