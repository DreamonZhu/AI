import pandas_code as pd

df = pd.read_csv("./IMDB-Movie-Data.csv")

# print(df.head())
# print(df.info())
#
# print(df['Metascore'].mean())
# print(df['Director'].head())

# 获取导演人数
# # expand=True 得到一个DataFrame
# print(df['Director'].str.split(expand=True))
# print(df['Director'].str.split(",").tolist())
# # 只有 Series 类型才可以 tolist()
# print(df['Director'].tolist())
# print(len(df['Director'].tolist()))
# print(len(df['Director'].unique().tolist()))

# 获取演员人数
# temp_actor_list = df['Actors'].str.split(",").tolist()
# actor_list = [i for j in temp_actor_list for i in j]
# print(len(actor_list))
# print(len(set(actor_list)))

# print(df.info())
# print(df['Runtime (Minutes)'].argmax())
# print(df['Runtime (Minutes)'].argmin())
print(df['Runtime (Minutes)'].idxmax())
print(df['Runtime (Minutes)'].idxmin())
print(df['Runtime (Minutes)'].median())
print(df['Runtime (Minutes)'].min())
print(df['Runtime (Minutes)'].max())
print(df['Runtime (Minutes)'][828])
print(df['Runtime (Minutes)'][793])

