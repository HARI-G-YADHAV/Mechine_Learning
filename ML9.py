import pandas as mypd
import matplotlib.pyplot as myplot
import seaborn as mysb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from IPython.display import display, HTML
import numpy as np

mydata = mypd.read_csv("/home/nasc/Documents/G/ML/doc/Wine.csv")
print (mydata)

"""
#pairploting
mysb.pairplot(mydata,hue ='quality')
myplot.show()
"""

mydata.isnull().sum()
for col in mydata.columns:
    if mydata[col].isnull().sum() > 0:
        mydata[col] = mydata[col].fillna(mydata[col].mean())
print(mydata.isnull().sum().sum())

x = mydata.iloc[:,9:11]
print(x)

y = mydata.iloc[:,12:13]
print(y)

mydata.hist(bins = 20, figsize = (10,10))
myplot.show()

myplot.bar(mydata['quality'],mydata['alcohol'])
myplot.xlabel('quality')
myplot.ylabel('alcohol')
myplot.show()

#correlation finding

myplot.figure(figsize = (12,12))
mysb.heatmap(mydata.corr(), annot = True)
myplot.show()

mydata.replace({'white' : 1,'red' : 0},inplace =True)
mydata['best quality'] = mydata.quality.apply(lambda x:1 if x > 5 else 0)
print(mydata['best quality'].value_counts())

from sklearn.model_selection import train_test_split
features = mydata.drop(['quality','best quality'],axis = 1)
target = mydata['best quality']
xtrain,xtest,ytrain,ytest = train_test_split(features, target, test_size = 0.2, random_state = 40, shuffle=True)
xtrain.shape,xtest.shape

mymodel = LogisticRegression()
mymodel = mymodel.fit(xtrain,ytrain)
print(mymodel)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
ypred = mymodel.predict(xtest)
model_acc = accuracy_score(ypred,ytest)
round(model_acc*100,2)

mymetrix = confusion_matrix(ypred,ytest)
print(mymetrix)

print(classification_report(ytest,ypred))




