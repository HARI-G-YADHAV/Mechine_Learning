import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import gc
import cv2

digits = pd.read_csv("/home/nasc/Documents/G/ML/doc/train.csv")
digits.info()

four = digits.iloc[3,1:]
four.shape

four = four.values.reshape(28,28)
plt.imshow(four,cmap = 'gray')

digits.label.astype('category').value_counts()

description = digits.describe()
print(description)
