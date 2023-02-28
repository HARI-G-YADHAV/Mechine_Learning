import pandas as mypd
import matplotlib.pyplot as myplot
import seaborn as mysb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
mydata = mypd.read_csv("/home/nasc/Documents/G/ML/doc/data.csv")
print(mydata,"\n")

x = mydata.iloc[:,1:2]
print(x)

y = mydata.iloc[:,2:3]
print("\n",y)

poly_reg = PolynomialFeatures(degree =4)
x_reg = poly_reg.fit_transform(x)
print("\n")

poly_reg.fit(x_reg,y)
lin2 = LinearRegression()
print(lin2.fit(x_reg, y))

myplot.scatter(x,y,color = 'blue')
myplot.plot(x, lin2.predict(poly_reg.fit_transform(x)),color = 'red')
myplot.xlabel('Temparature')
myplot.title('Polynomial Regeression')
myplot.ylabel('Pressure')
myplot.show()