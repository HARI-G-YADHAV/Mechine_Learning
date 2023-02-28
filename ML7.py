#one hot encoding
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


poke_df = pd.read_csv("/home/nasc/Documents/G/ML/pokemon.csv",encoding = 'utf-8')

#trasform and map pokemon generation
gen_le = LabelEncoder()
gen_labels = gen_le.fit_transform(poke_df['Generation'])
poke_df['Gen_Label'] = gen_labels

#transform and map pokemon legendary status
leg_le = LabelEncoder()
leg_Labels = leg_le.fit_transform(poke_df['Legendary'])
poke_df['Lgnd_label'] = leg_Labels

poke_df_sub = poke_df[['Name','Generation','Gen_Label','Legendary','Lgnd_label']]
print(poke_df_sub.iloc[0:40])

#catogory transform

gen_onehot_features = pd.get_dummies(poke_df['Generation'])
pd.concat([poke_df[['Name','Generation']], gen_onehot_features], axis = 1).iloc[4.10]


