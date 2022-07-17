
# from google.colab import drive
# drive.mount('/content/drive')
import pandas as pd
from sklearn import svm
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split

print(len(os.listdir('/Users/bibinkunjumon/Downloads/Programs/data/cat')))

print(len(os.listdir('/Users/bibinkunjumon/Downloads/Programs/data/dog')))

catpath = os.path.join('/Users/bibinkunjumon/Downloads/Programs/data','cat')
print(catpath)

for img in os.listdir(catpath):
    print(img)

flat_data_arr = []
target_arr = []
categories = ['cat','dog']
data_dir = '/Users/bibinkunjumon/Downloads/Programs/data'
