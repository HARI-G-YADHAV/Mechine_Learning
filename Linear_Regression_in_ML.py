import pandas as mypd
import matplotlib.pyplot as myplot
import seaborn as mysb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

mydata = mypd.read_csv("/home/nasc/Documents/G/ML/Mult_Reg_Yield.csv")
print(mydata)

#to check missing values
print("\n",mydata.describe())
print("\n",mydata.info())

x = mydata.iloc[:,0:2]
print(x)

y = mydata.iloc[:,2:3]
print("\n",y)

"""
#Correlation Analysys
#scatter plot
mysb.pairplot(mydata)
myplot.show()"""

#Regresssion  modelin
#fitting the model
mymodel = LinearRegression()
print(mymodel)

mymodel = mymodel.fit(x,y)
print(mymodel)

pred = mymodel.predict(x)
print(pred)

print("\n",mymodel.coef_)

print("\n",mymodel.intercept_)

rsq = mymodel.score(x,y)
print("\n",rsq)

print("\n",round(rsq*100,2))

mse = mean_squared_error(y,pred)
print("\n",mse)

import math as mymath
rmse = mymath.sqrt(mse)
print(rmse)

#Residual analysis
res = y-pred
print(res)
