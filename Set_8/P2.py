
import pandas as pd
from sklearn.svm import SVC
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay


# ------- To create dataframe from image files -------

flat_data_arr = []
target_arr = []
categories = ['cat','dog']
data_dir = '/Users/bibinkunjumon/Downloads/Programs/data'

for i in categories:
    print("Start Work Category :",i)
    path_1 = os.path.join(data_dir,i)
    for img in os.listdir(path_1):
        img_array = imread(os.path.join(path_1,img))
        img_resize = resize(img_array,(150,150,3))
        flat_data_arr.append(img_resize.flatten())
        target_arr.append(categories.index(i))
    print("Work finished",i)

# print(target_arr) o/p 0 & 1
# print(flat_data_arr) o/p linear array of image data
# converting into numpy array
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
# forming dataframe
df = pd.DataFrame(flat_data)  # no of pixels becomes columns & rows= cat+dog pic count
df['Target']=target # adding target column to df
print("*"*100,flat_data)
print("*"*100,df)
print(df.columns)

# -------- END    :         Here we got Dataframe to process --------

X = df.drop('Target',axis=1)
y = df['Target']

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.70,random_state=1)

model = SVC()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
#plt.show()

# ---------- Here I am testing with a random image downloaded from google and predicting Dog or Cat
path_test = '/Users/bibinkunjumon/Downloads/Programs/data/dog_test.jpeg'
img_test = imread(path_test)
print(img_test)
img_test = resize(img_test,(150,150,3)).flatten().reshape(1,-1)  # reshape 1,-1 : Single row -1,1 : Single column
print(model.predict(img_test))  # I get out put 1 ie: Dog... SUCCESS..!!!!!!!

# ------------- END