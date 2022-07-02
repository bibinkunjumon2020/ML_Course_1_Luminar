
# ------- Naive Bayes - MultinomialNB() algorithm : weather data set
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

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
play_enc = LabelEncoder().fit_transform(play)

#---- create input set

X = list(zip(weather_enc,temp_enc))
print(X)

#---naive bayers model

model = MultinomialNB()
# train model with full input set and output labels
model.fit(X,play_enc)
# predicting output
predict1 = model.predict([[2,1]])  #Sunny , Hot ->op must be No but we ger YEs
print("Prediction - ",predict1)
predict2 = model.predict([[0,2]]) #Overcast,mild op is Yes.Its right
print(predict2)

"""
Here we used entire input to train and random input for testing the model.
Accuracy cannot be determined since no test output set

"""

