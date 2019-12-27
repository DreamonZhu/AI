import pandas_code as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("./IMDB-Movie-Data.csv")

print(df.info())
genre = df['Genre']

temp_genre_list = genre.str.split(",").tolist()
genre_list = [i for j in temp_genre_list for i in j]
genre_nums = len(set(genre_list))

zero_df = pd.DataFrame(np.zeros((genre.shape[0], genre_nums)), columns=set(genre_list))
# print(zero_df)

for i in range(genre.shape[0]):
    zero_df.loc[i, temp_genre_list[i]] = 1

print(zero_df.head())

genre_count = zero_df.sum(axis=0)
print(genre_count)

genre_count = genre_count.sort_values()
_x = genre_count.index
_y = genre_count.values

plt.figure(figsize=(20, 8), dpi=80)
plt.bar(range(len(_x)), _y, width=0.4, color="pink")
plt.xticks(range(len(_x)), _x)
plt.show()
