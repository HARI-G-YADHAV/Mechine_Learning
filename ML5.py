#nominal Encoding

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing

#load the data set
df = pd.read_csv("cluster_mpg.csv")

print(df.head())

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df["origin"])

print(list(label_encoder.classes_))

print(label_encoder.transform(df["origin"]))


