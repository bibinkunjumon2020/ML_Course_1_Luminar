
# Binary Classification Problem -: 0/1 is label Or Yes/No or only 2 groups

import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#------Importing Fila as DataFrame
df=pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/diabetes.csv")
# Check for null values - if there remove it or fill it
print(df.isna().sum())
#print(df.head())
print("**********")
print("Shape=",df.shape)
print("Size = ",df.size)

#----- Creating train and test inputs,outputs
input_set=df.iloc[:,:-1].values  #iloc row 0 to 0 Column 0 to -1(-1 wont executed ) ie: last column not read
output_set=df.iloc[:,-1].values # last column all rows
#print(output_set)
train_input,test_input,train_output,test_output=tts(input_set,output_set,train_size=0.75)
#I can assign test size or train size test_size=.30 : always train set big

#----- Scaling the input sets

scalar=SS()
scalar.fit(train_input)
train_input=scalar.transform(train_input)
test_input=scalar.transform(test_input)

#----fitting knn and predicting

knn=KNC(n_neighbors=301) #5,11,51,101 any value i can give here for making model
knn.fit(train_input,train_output)
# prediction
prediction=knn.predict(test_input) # This is the prediction output of my Model
#print(prediction)

#----- random checking based on my model to Check accuracy
    # Here I need to give all the given features in the DF to check with real output
print(knn.predict([[6,148,72,35,0,33.6,0.627,50]])) #I got result same as source


#------ Accuracy Checking of my model

con_matrix=confusion_matrix(test_output,prediction)
score_accuracy=accuracy_score(test_output,prediction)

print("Confusion Matrix is : \n ",con_matrix)
print("My Model Accuracy :  ",score_accuracy)

# -----------
'''
True Negative - hw many 0s conveted as 0 : 0->0
True Positive -  hw many 1s converted as 1 :1->1

False Negative - 1-> 0 (check o/p for negative and positive
False Positive - 0-> 1
matrix:
                          Actual
                    Positive     Negative  
 Predicted :   +     TP             FP                  ::::: THis is the confusion Matrix
               -     FN             TN
 
Accuracy = (TP+TN)/ (TP+TN+FP+FN)

CM : [TP,FP]
     [FN,TN]


'''