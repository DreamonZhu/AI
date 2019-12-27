import pandas_code as pd

df = pd.read_csv("./starbucks_store_worldwide.csv")
print(df.info())
# print(df.head())
# print(df.tail())

# 方法一
# A_df = df[df['Country'] == 'US']
# C_df = df[df['Country'] == 'CN']
# print(type(A_df))
# print(A_df.shape[0])
# print(C_df.shape[0])
# print(len(A_df))
# print(len(C_df))

# 方法二:聚合
grouped = df.groupby(by="Country")
# print(grouped)
# grouped: (分类的名称（在本例中是国家）：DataFrame) (US: DataFrame)、(ZA: DataFrame)
# for i, j in grouped:
#     print(i)
#     print(j, type(j))

country_count = grouped.count()['Brand']
print(type(country_count))

# DataFrame 类型数据
# print(type(df[['Brand']]))
# print(df[['Brand']])
# Series 类型数据
# print(df['Brand'])

grouped1 = df[["Brand"]].groupby(by=[df['Country'], df['State/Province']]).count()
grouped2 = df.groupby(by=['Country', 'State/Province'])[["Brand"]].count()
grouped3 = df.groupby(by=['Country', 'State/Province'])["Brand"].count()

# print(type(grouped1))
# print(type(grouped2))
# print(type(df))
#
# for i, j in grouped1:
#     print(i)
#     print(j)
print(grouped1)
print(grouped2)
print(grouped3)
print(grouped1.loc['CN', '11'])
print(grouped1.swaplevel().loc['11'])
# print(grouped1.values)
# print(grouped1.swaplevel().loc['BJ'])
