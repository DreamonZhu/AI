import pandas_code as pd
from matplotlib import pyplot as plt

df = pd.read_csv('./starbucks_store_worldwide.csv')
data = df.groupby(by=['Country', 'City']).count().loc['CN','Brand']
# data = df.groupby(by=['Country', 'City'])[['Brand']]
# for i, j in data:
#     print(j)
print(data)
