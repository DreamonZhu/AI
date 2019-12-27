import pandas_code as pd

df = pd.read_csv('./books.csv')
print(df.head())
print(df.info())

data = df.groupby(by="original_publication_year")["average_rating"]
print(data.count())

print(data.mean())
