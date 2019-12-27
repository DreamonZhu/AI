import pandas_code as pd
from matplotlib import pyplot as plt

df = pd.read_csv('./IMDB-Movie-Data.csv')
# print(df.head())
# print(df.info())
Rating = df['Rating']
# print(Rating.max())
# print(Rating.min())

plt.figure(figsize=(20, 8), dpi=80)

bin_width = 0.5
bin_nums = (Rating.max() - Rating.min()) // bin_width
plt.hist(Rating, int(bin_nums))

_x = [Rating.min()]
i = Rating.min()
while i < Rating.max() + 0.5:
    i += 0.5
    _x.append(i)

print(_x)
plt.xticks(_x)
plt.show()
