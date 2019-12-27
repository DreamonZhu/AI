import pandas_code as pd

# 在字典中的v存在一个value值已经自身存在索引时，可以不再传索引，否则必须传
a = pd.DataFrame({'a': range(7),'b': range(7, 0, -1),'c': ['one','one','one','two','two','two', 'two'],'d': list("hjklmno")})
