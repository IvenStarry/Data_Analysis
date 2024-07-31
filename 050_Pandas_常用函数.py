import pandas as pd
import numpy as np
'''
读取数据

pd.read_csv(filename)	                读取 CSV 文件
pd.read_excel(filename)	                读取 Excel 文件
pd.read_sql(query, connection_object)	从 SQL 数据库读取数据
pd.read_json(json_string)	            从 JSON 字符串中读取数据
pd.read_html(url)	                    从 HTML 页面中读取数据
'''
# # 读取 CSV 文件
# df = pd.read_csv('data.csv')
# # 读取 Excel 文件
# df = pd.read_excel('data.xlsx')
# # 从 SQL 数据库读取数据
# import sqlite3
# conn = sqlite3.connect('database.db')
# df = pd.read_sql('SELECT * FROM table_name', conn)
# # 从 JSON 字符串中读取数据
# json_string = '{"name":"john", "age":30, "city":"chengdu"}'
# df = pd.read_json(json_string)
# # 从 HTML 页面中读取数据
# url = 'https://github.com/IvenStarry'
# dfs = pd.read_html(url)
# df = dfs[0]
# print(df)

# '''
# 查看数据

# df.head(n)	    显示前 n 行数据
# df.tail(n)	    显示后 n 行数据
# df.info()	    显示数据的信息，包括列名、数据类型、缺失值等
# df.describe()	显示数据的基本统计信息，包括均值、方差、最大值、最小值等
# df.shape    	显示数据的行数和列数
# '''
# print(df.head())
# print(df.tail())
# print(df.info())
# print(df.describe())
# print(df.shape)

'''
数据清洗

df.dropna()	                        删除包含缺失值的行或列
df.fillna(value)	                将缺失值替换为指定的值
df.replace(old_value, new_value)	将指定值替换为新值
df.duplicated()	                    检查是否有重复的数据
df.drop_duplicates()	            删除重复的数据
'''

'''
数据选择和切片

df[column_name]	                                选择指定的列；
df.loc[row_index, column_name]	                通过标签选择数据；
df.iloc[row_index, column_index]	            通过位置选择数据；
df.ix[row_index, column_name]	                通过标签或位置选择数据；
df.filter(items=[column_name1, column_name2])	选择指定的列；
df.filter(regex='regex')	                    选择列名匹配正则表达式的列；
df.sample(n)	                                随机选择 n 行数据。
'''

'''
数组排序

df.sort_values(column_name)	                                            按照指定列的值排序
df.sort_values([column_name1, column_name2], ascending=[True, False])	按照多个列的值排序
Series.sort_values(axis=0, ascending=True)
DataFrame.sort_values(by, axis=0, ascending=True)                       by axis轴上某个索引或索引列表
df.sort_index(axis=0, ascending=True)	                                按照索引排序
'''
a = pd.DataFrame(np.arange(20).reshape(4, 5), index=['c', 'a', 'd', 'b'])
print(a)
print(a.sort_index())
print(a.sort_index(ascending=False))
print(a.sort_index(axis=1, ascending=False))
print(a.sort_values(2, ascending=False))
print(a.sort_values('a', axis=1, ascending=False))

# 若有NAN 排序统一置于末尾
a = pd.DataFrame(np.arange(20).reshape(4, 5), index=['c', 'a', 'd', 'b'])
b = pd.DataFrame(np.arange(12).reshape(3, 4), index=['a', 'b', 'c'])
print(a + b)
print((a+b).sort_values(2))
print((a+b).sort_values(2, ascending=False))

'''
数组的分组和聚合

df.groupby(column_name)                        	 按照指定列进行分组；
df.aggregate(function_name)	                     对分组后的数据进行聚合操作；
df.pivot_table(values, index, columns, aggfunc)	 生成透视表。
'''

'''
数据合并

pd.concat([df1, df2])	将多个数据框按照行或列进行合并；
pd.merge(df1, df2, on=column_name)	按照指定列将两个数据框进行合并。
'''

'''
数据选择和过滤

df.loc[row_indexer, column_indexer]	    按标签选择行和列。
df.iloc[row_indexer, column_indexer]	按位置选择行和列。
df[df['column_name'] > value]	        选择列中满足条件的行。
df.query('column_name > value')	        使用字符串表达式选择列中满足条件的行。
'''

'''
数据统计和描述

df.describe()	计算基本统计信息，如均值、标准差、最小值、最大值等。
df.mean()	    计算每列的平均值。
df.median()	    计算每列的中位数。
df.mode()	    计算每列的众数。
df.count()	    计算每列非缺失值的数量。
.argmin()       计算数据最大值的索引位置(自动索引) 或被弃用
.argmax()       计算数据最小值的索引位置(自动索引)
.idxmin()       计算数据最大值的索引(自定义索引)
.idxmax()       计算数据最小值的索引(自定义索引)
'''

df = pd.read_json('related_data/data.json')
print(df)
# df = df.dropna()
# print(df)
df = df.fillna({'age':0, 'score':0})
print(df)
# 重命名列名
df = df.rename(columns={'name':'姓名', 'age':'年龄', 'gender':'性别', 'score':'成绩'})
print(df)
# 按成绩排序
df = df.sort_values(by='成绩', ascending=False)
print(df)
# 按性别分组计算平均年龄和成绩
grouped = df.groupby('性别').agg({'年龄':'mean', '成绩':'mean'}) # df.aggregate(function_name) 在这个例子中，对 年龄和成绩 应用了 mean 函数
print(grouped)
# 选择成绩大于90的行，并只保留姓名和成绩两列
sorted = df.loc[df['成绩'] >= 90, ['姓名' , '成绩']]
print(sorted)
print(df.count())
print(df.idxmax())

'''
累计统计分析函数

.cumsum()             前n个数的和
.cumprod()            前n个数的积
.cummax()             前n个数的最大值
.cummin()             前n个数的最小值

.rolling(w).sum()     依次计算相邻w个元素的和
.rolling(w).mean()    依次计算相邻w个元素的算术平均值
.rolling(w).var()     依次计算相邻w个元素的方差
.rolling(w).std()     依次计算相邻w个元素的标准差
.rolling(w).min()     依次计算相邻w个元素的最小值
.rolling(w).max()     依次计算相邻w个元素的最大值
'''

a = pd.DataFrame(np.arange(20).reshape(4, 5), index=['c', 'a', 'd', 'b'])
print(a)
print(a.cumsum() )
print(a.cumprod())
print(a.cummax() )
print(a.cummin() )
print(a.rolling(2).sum()) # 计算包括自己的前两项和 第一列前面没有数值 返回NAN
print(a.rolling(2).mean())
print(a.rolling(2).var())
print(a.rolling(2).std())
print(a.rolling(2).min())
print(a.rolling(2).max()) 