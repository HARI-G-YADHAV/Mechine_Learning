import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from sklearn import preprocessing

df = pd.read_csv("/home/nasc/Documents/G/ML/doc/data.csv")
print(df.head())

df = df.drop('id',axis = 1)
df = df.drop('Unnamed: 32',axis = 1)
#Mapping Benign toi 0 and Malignant To 1
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
#scaling dataset
datas = pd.DataFrame(preprocessing.scale(df.iloc[:,1:32]))
datas.columns = list(df.iloc[:,1:32].columns)
datas['diagnosis']  =df['diagnosis']
#Creating the high dimentional feature space in x
data_drop = datas.drop('diagnosis',axis = 1)
x = data_drop.values

#creating 2D visualisation to visualize the cluster
from sklearn.manifold import TSNE
tsne = TSNE(verbose=1,perplexity=40,n_iter= 4000)
y = tsne.fit_transform(x)

from sklearn.cluster import KMeans
kmns = KMeans(n_clusters=2,init='k-means++',n_init=50,max_iter=300,tol=0.0001)
kY = kmns.fit_predict(x)

f,(ax1,ax2) = plt.subplots(1,2,sharey=True)
ax1.scatter(y[:,0],y[:,1], c=kY,cmap = "jet",edgecolors="None",alpha=0.35)
ax1.set_title('k-means clustering point')

ax2.scatter(y[:,0],y[:,1], c=datas['diagnosis'],cmap = "jet",edgecolors="None",alpha=0.35)
ax2.set_title('actual clusters')

from sklearn_extra.cluster import KMedoids
KMedoids = KMedoids(n_clusters=2,random_state=0,max_iter=300)
kY = kmns.fit_predict(x)

f,(ax1,ax2) = plt.subplots(1,2,sharey=True)
ax1.scatter(y[:,0],y[:,1], c=kY,cmap = "jet",edgecolors="None",alpha=0.35)
ax1.set_title('k-Medoid clustering point')

ax2.scatter(y[:,0],y[:,1], c=datas['diagnosis'],cmap = "jet",edgecolors="None",alpha=0.35)
ax2.set_title('actual clusters')

#hierarchical clustering

from sklearn.cluster import AgglomerativeClustering
aggc = AgglomerativeClustering(n_clusters = 2,linkage='ward')
ky = aggc.fit_predict(x)
f, (ax1,ax2) = plt.subplots(1,2,sharey = True)

ax1.scatter(y[:,0],y[:,1], c =ky, cmap = 'jet', edgecolor = 'None', alpha = 0.35)
ax1.set_title('hierarchical clustering plot')

ax1.scatter(y[:,0],y[:,1], c =datas['diagnosis'], cmap = 'jet', edgecolor = 'None', alpha = 0.35)
ax2.set_title('Actual clusteing')
