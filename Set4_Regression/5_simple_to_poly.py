
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df = pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/book4.csv")
print(df.head())
#print(df.count())

# --Cleaning data

X = df.drop(['Pressure','Sno'],axis=1)  # Warning :LR only 1 feature column 1 Label column
y = df['Pressure']
#print(X)
#print(y)

# --- model fitting

model = LinearRegression()
model.fit(X,y)

# -- Prediction

y_pred = model.predict(X)
print(y_pred)

# -- Plotting

plt.scatter(X,y,color='red')
plt.plot(X,y_pred,color='blue')
plt.title("SLR")
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()

'''
From output graph its clear that y_pred and y actual has no relation.So no linear regression method possible here.
So we have to adopt Polynomial Regression method
c=y+mx --> d=C+Bx+Ax^2
'''

