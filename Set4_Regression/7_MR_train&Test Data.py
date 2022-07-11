import pandas as pd
from sklearn.linear_model import LinearRegression

# ------------------------------ Train Data -----------------
# - reading train data set
df = pd.read_csv("/Source/train-data.csv")
print(df.columns)

# -- Data cleaning

print(df['Location'].value_counts())
df_dum = pd.get_dummies(df[['Location',"Fuel_Type",'Transmission','Owner_Type','Name']],drop_first=True)
print(df_dum.columns)
df.drop(['Location','Fuel_Type','Transmission','Owner_Type','Name','New_Price'],axis=1,inplace=True)
df=df.iloc[:,1:]   # removed first column
print(df.head())
print(df.columns)
print(pd.isna(df).sum())

# -- Unnecessary Units removal
df['Mileage'] = df['Mileage'].str.replace('kmpl','')
df['Mileage'] = df['Mileage'].str.replace('km/kg','')
df['Engine'] = df['Engine'].str.replace('CC','')
df['Power'] = df['Power'].str.replace('bhp','')

print(df.head(10))
# -- Checking types
print(df.dtypes)
# Conerting objects and strings to float

df['Mileage']=df['Mileage'].astype(float)
df['Engine'] = df['Engine'].astype(float)
df['Power'] = df['Engine'].astype(float)
print(df.dtypes)

# -----------Filling null points with mean in case of float and mode in case of discrete values

df['Seats']=df['Seats'].fillna(df['Seats'].mode()[0]) # Here [0] after mode is must..otherwise a failure
df['Engine']=df['Engine'].fillna(df['Engine'].mean())
df['Power'] = df['Power'].fillna(df['Engine'].mean())
df['Mileage'] = df['Mileage'].fillna(df['Engine'].mean())
print(pd.isna(df).sum())  # -- No nulls

#-- attaching dummy and df into single data frame

df =pd.concat([df,df_dum],axis=1)

# ------- At this point all data required retained discarded unnecessary,converted to decimal and float
# filled null with mean and mode

# Separating X and y
X_train = df.drop(['Price'],axis=1)
y = df['Price']

# Multiple Linear Regression

model = LinearRegression()
model.fit(X_train,y)

# --- This data will be used for Test data

# ------------------------------ Test Data -----------------
# - reading train data set
df = pd.read_csv("/Source/test-data.csv")
print(df.columns)

# -- Data cleaning

print(df['Location'].value_counts())
df_dum = pd.get_dummies(df[['Location',"Fuel_Type",'Transmission','Owner_Type','Name']],drop_first=True)
print(df_dum.columns)
df.drop(['Location','Fuel_Type','Transmission','Owner_Type','Name','New_Price'],axis=1,inplace=True)
df=df.iloc[:,1:]   # removed first column
print(df.head())
print(df.columns)
print(pd.isna(df).sum())

# -- Unnecessary Units removal
df['Mileage'] = df['Mileage'].str.replace('kmpl','')
df['Mileage'] = df['Mileage'].str.replace('km/kg','')
df['Engine'] = df['Engine'].str.replace('CC','')
df['Power'] = df['Power'].str.replace('bhp','')

print(df.head(10))
# -- Checking types
print(df.dtypes)
# Conerting objects and strings to float

df['Mileage']=df['Mileage'].astype(float)
df['Engine'] = df['Engine'].astype(float)
df['Power'] = df['Engine'].astype(float)
print(df.dtypes)

# -----------Filling null points with mean in case of float and mode in case of discrete values

df['Seats']=df['Seats'].fillna(df['Seats'].mode()[0]) # Here [0] after mode is must..otherwise a failure
df['Engine']=df['Engine'].fillna(df['Engine'].mean())
df['Power'] = df['Power'].fillna(df['Engine'].mean())
df['Mileage'] = df['Mileage'].fillna(df['Engine'].mean())
print(pd.isna(df).sum())  # -- No nulls

#-- attaching dummy and df into single data frame

X_test =pd.concat([df,df_dum],axis=1) # Here the whole set is X_test


# -------------------- Predicting --------------------------------
print("-------------------- Predicting --------------------------------")

y_pred = model.predict(X_test)
print(y_pred)





