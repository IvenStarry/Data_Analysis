import pandas as pd

'''
CSV comma separated values 逗号分隔值 以纯文本形式存储表格数据
是一种通用的、箱底简单的文件格式
pandas可以处理CSV文件
'''

# 读取CSV文件 read_csv
df = pd.read_csv('related_data/nba.csv')
print(df.to_string()) # to_string可以返回dataframe类型的数据，若不使用，数据只显示前五行和后五行，中间用...代替
print(df)

# 存储CSV文件 to_csv
name = ['Iven', 'Rosenn', 'Starry']
age = [19, 20, 21]
hobby = ['food', 'coding', 'travel']
dict = {'name':name, 'age':age, 'hobby':hobby}
df = pd.DataFrame(dict)
df.to_csv('related_data/savetest.csv')

# 数据处理 
df = pd.read_csv('related_data/nba.csv')
# head(n) 读取前面n行，默认5行
print(df.head(3))
# tail(n) 读取尾部n行，默认5行
print(df.tail(3))
# info() 返回表格的一些基本信息
print(df.info())