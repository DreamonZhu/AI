import pandas_code as pd
from matplotlib import pyplot as plt

df = pd.read_csv('./starbucks_store_worldwide.csv')
print(df.info())
print(df.tail())

store_num = df.groupby(by="Country").count()['Brand'].sort_values(ascending=False)[:10]
print(store_num)

plt.figure(figsize=(20, 8), dpi=80)

plt.bar(range(store_num.shape[0]), store_num, width=0.4, color="pink")
plt.xticks(range(store_num.shape[0]), store_num.index)

plt.show()
