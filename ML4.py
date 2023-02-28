import category_encoders as ce
import pandas  as pd

df1 = pd.DataFrame({'height': ['tall','medium','short','tall','medium','short','tall','medium']})

encoder = ce.OrdinalEncoder(cols = ['height'],return_df = True,mapping = [{'col':'height','mapping' : {'None':0,'tall':1,'medium':2,'short':3}}])
#original data

print(df1,"\n")
df1['transformed'] = encoder.fit_transform(df1)
print(df1)
