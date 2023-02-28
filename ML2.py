import pandas as pd
import matplotlib.pyplot as plt
dict = {'DATE':[31-1-2020,29-2-2020,30-3-2020,31-3-2020,30-5-2020],
        'PRICE':[10000,20000,30000,40000,50000],
        'PRODUCT_ID':[1,2,3,4,5],
        'QUANTITY_PURCHASE':[30,31,32,33,34],
        'SERAIL_NO':[100,101,102,103,104],
        'USR_ID':[1001,1002,1003,1004,1005],
        'USR_TYPE':['A','B','C','D','E'],
        'USR_CLASS':['UPR','UPR','LWR','UPR','MDL'],
        'PUR_WEEK':['MON','WED','THU','MON','WED']}
df = pd.DataFrame(dict)
df.to_csv("df.csv")
print(df)
#print all statistics
stats = df.describe(include = 'all')
print(stats)
#plot 
df.plot(x ="USR_ID",y = "PRICE",kind = "line")
#plt.show()
df[['QUANTITY_PURCHASE','PUR_WEEK']].plot.box()
plt.title('Quantity and week value distribution')
plt.show()

