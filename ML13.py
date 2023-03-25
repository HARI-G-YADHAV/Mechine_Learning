import pandas as pd
df = pd.read_csv("/home/nasc/Documents/G/Mechine_Learning/doc/Wine.csv")

print(df[['type','citric acid','alcohol']].head())

print(df.replace({'white' : 0,'red' : 1},inplace =True))
