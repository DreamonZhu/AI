from matplotlib import pyplot as plt
import pandas_code as pd

df = pd.read_csv("./IMDB-Movie-Data.csv")
print(df.info())
runtime_data = df['Runtime (Minutes)']

print(runtime_data.max())
print(runtime_data.min())
bin_nums = (runtime_data.max() - runtime_data.min()) // 5

plt.figure(figsize=(20, 8), dpi=80)
plt.hist(runtime_data, bin_nums)
i = runtime_data.min()
_x = [runtime_data.min()]
while i < runtime_data.max()+5:
    i += 5
    _x.append(i)

print(_x)
plt.xticks(_x)
plt.grid(True, linestyle="-.")
plt.show()
