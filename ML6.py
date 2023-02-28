#ordinal Encording

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
vg_df = pd.read_csv("/home/nasc/Documents/G/ML/vgsales.csv",encoding = "utf-8")
print(vg_df[["Name","Platform","Year","Genre","Publisher"]].iloc[1:7])
genres = np.unique(vg_df["Genre"])
print("\n",genres,"\n")
gle = LabelEncoder()
genre_Labels = gle.fit_transform(vg_df['Genre'])
genre_mappiings = {index: label for index,label in enumerate(gle.classes_)} 
print("\n",genre_mappiings,"\n")
vg_df['GenreLabel'] = genre_Labels
print(vg_df[["Name","Platform","Year","Genre","GenreLabel"]].iloc[1:7])






