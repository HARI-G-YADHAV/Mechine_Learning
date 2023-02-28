#gaussian model finding probabilty
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mydata = pd.read_csv("/home/nasc/Documents/G/ML/doc/Iris_data.csv")
print(mydata.head())

x = mydata.iloc[:,:4]
y = mydata['Species'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
ypred = classifier.predict(x_test)
mymetrix = confusion_matrix(ypred,y_test)
print(mymetrix)
print(classification_report(y_test,ypred))

acc = accuracy_score(ypred,y_test)
print(acc)




