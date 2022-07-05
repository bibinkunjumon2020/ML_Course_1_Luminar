import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("https://raw.githubusercontent.com/bibinkunjumon2020/ML_Course_1_Luminar/master/50_Startups.csv",on_bad_lines='skip')
"""
on_bad_lines :
Lines with too many fields (e.g. a csv line with too many commas) will by default cause an exception to be raised, 
and no DataFrame will be returned. 
If False, then these “bad lines” will be dropped from the DataFrame that is returned.
"""
#print(df.head())
#print(df.info())
#print("Value Counts =\n",df['State'].value_counts())
#print("Uniques : \n",df['State'].unique())

# x & y
X = df.drop(['Profit'],axis=1)
#print(X)
y = df['Profit']
#print(y)

#Preprocessing
col_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'),['State']),remainder='passthrough')
# Here Onehot encoder followed by the column that needs to be encoded as number from string,and also this
# transform is better than other bcs its unbiased (1,0,0)
# We can use label encoder also for converting string to numeric.
#but label encoder is biased
#handle unknown - for handling unexpected value in test data which only present in train data
#remainder passthrough allows other column values also encoded along with

X = col_trans.fit_transform(X)
print(X)

# -- Comparing each feature output relation using seaborn module
x_axis = df['R&D Spend']
y_axis=y
sns.regplot(x_axis,y_axis,color='red')
#plt.show()

sns.regplot(df['Administration'],y)
#plt.show()

sns.regplot(df['Marketing Spend'],y)
#plt.show()

# ---- data process

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.75,random_state=1)

# --model
model = LinearRegression()
model.fit(X_train,y_train)

# -- predict
y_pred = model.predict(X_test)

print("Predictions ------- \n",y_pred)

r2 = r2_score(y_test,y_pred)
print("r2 score = ",r2)

# Data frame shows some values for comparison

df = pd.DataFrame({'Actual Value ':y_test,'Predicted Value':y_pred,'Diff':(y_test-y_pred)})
print(df)
plt.show()   # all 3 seaborn shows in one graph