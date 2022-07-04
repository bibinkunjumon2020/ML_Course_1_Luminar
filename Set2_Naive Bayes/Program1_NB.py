
# ------- Naive Bayes - MultinomialNB() algorithm : weather data set
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
# list of datas initialized here as lists (No file processing,No dataset import)

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No',]
#print(len(weather))
#print(len(temp))
#print(len(play))

#------ Encoding each label using preprocessing lib

weather_enc = LabelEncoder().fit_transform(weather)
temp_enc = LabelEncoder().fit_transform(temp)
#y = LabelEncoder().fit_transform(play)
y=play     # to get Yes/No output i am avoiding transforming y
#---- create input set

X = list(zip(weather_enc,temp_enc))
print("Features :\n",X)
# DATA SET OUTPUT CHECKING - balanced or imbalanced.IF binary output has similar o/p count,then Balanced,otherwise imbalanced
# depends on counter output we can decide which algorithm is better
print(Counter(y))
#---naive bayers model

model = MultinomialNB()
# train model with full input set and output labels
model.fit(X,y)
# predicting output
predict1 = model.predict([[2,1]])  #Sunny , Hot ->op must be No but we ger YEs
print("Prediction 1 - ",predict1)
predict2 = model.predict([[0,2]]) #Overcast,mild op is Yes.Its right
print("Prediction 2 - ",predict2)

"""
Here we used entire input to train and random input for testing the model.
Accuracy cannot be determined since no test output set

"""

