import pandas as pd

'''
数据清洗是对一些没有用的数据进行处理的过程
很多数据集存在数据缺失、数据格式错误或重复数据的情况，为了使数据分析更加准确，就需要对这些数据进行处理
有四种空数据：n/a NA -- na
'''

# 清洗空值 
'''
dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

axis：默认为 0，表示逢空值剔除整行，如果设置参数 axis＝1 表示逢空值去掉整列。
how：默认为 'any' 如果一行（或一列）里任何一个数据有出现 NA 就去掉整行，如果设置 how='all' 一行（或列）都是 NA 才去掉这整行。
thresh：设置需要多少非空值的数据才可以保留下来的。
subset：设置想要检查的列。如果是多个列，可以使用列名的 list 作为参数。
inplace：如果设置 True，将计算得到的值直接覆盖之前的值并返回 None，修改的是源数据。

isnull() 判断各个单元格是否为空
'''

df = pd.read_csv('related_data/property-data.csv')
print(df.to_string())
print(df['NUM_BEDROOMS'])
print(df['NUM_BEDROOMS'].isnull()) # na是空数据但没被判为空数据

# 指定空数据类型 na_values
missing_values = ['n/a', 'na', '--']
df = pd.read_csv('related_data/property-data.csv', na_values=missing_values)
print(df['NUM_BEDROOMS'])
print(df['NUM_BEDROOMS'].isnull())

# 删除空数据的行 默认返回新的DataFrame
new_df = df.dropna()
print(new_df.to_string())
df.dropna(inplace=True) # 直接修改原数据
print(df.to_string())

# 指定列有空值的行
df = pd.read_csv('related_data/property-data.csv')
df.dropna(subset=['ST_NUM'], inplace=True)
print(df.to_string())

# fillna() 替换一些空字段
df = pd.read_csv('related_data/property-data.csv')
df.fillna(12345, inplace=True)
print(df.to_string())

# 指定某一列来替换数据
df = pd.read_csv('related_data/property-data.csv')
df['PID'].fillna(12345, inplace=True)
print(df.to_string())

# 替换空单元格的常用方法是计算列的均值、中位数或众数
df = pd.read_csv('related_data/property-data.csv')
x = df['ST_NUM'].mean() # mean() 计算均值
df_mean = df['ST_NUM'].fillna(x)
print(df_mean.to_string())

x = df['ST_NUM'].median() # median() 计算中位数
df_median = df['ST_NUM'].fillna(x)
print(df_median.to_string())

x = df['ST_NUM'].mode() # mode() 计算众数
df_mode = df['ST_NUM'].fillna(x)
print(df_mode.to_string())

# 清洗格式错误数据
# 通过包含空单元格的行，或者将列中的所有单元格转换为相同格式的数据
data = {
    'Date':['2020/12/01', '2020/12/02', '20201226'],
    'duration':[50, 40, 45]
}
df = pd.DataFrame(data, index=['day1', 'day2', 'day3'])
df['Date'] = pd.to_datetime(df['Date'], format='mixed') # to_datetime格式化日期
print(df.to_string())

data = {
    'Date':['2020/12/01', '2020/12/02', '20201226'],
    'duration':[50, 40, 12345] # 12345 数据是错误的
}
df = pd.DataFrame(data)
df.loc[2, 'duration'] = 30 # 修改错误日期  loc(row_index, column_index)
print(df.to_string())

data = {
    'Date':['2020/12/01', '2020/12/02', '20201226'],
    'duration':[50, 40, 12345] # 12345 数据是错误的
}
df = pd.DataFrame(data)
for x in df.index:
    if df.loc[x, 'duration'] > 120: # 设置条件语句将大于120的值设为120
        df.loc[x, 'duration'] = 120
print(df.to_string())

data = {
    'Date':['2020/12/01', '2020/12/02', '20201226'],
    'duration':[50, 40, 12345] # 12345 数据是错误的
}
df = pd.DataFrame(data)
for x in df.index:
    if df.loc[x, 'duration'] > 120: # 将错误数据的行删除
        df.drop(x, inplace=True)
print(df.to_string())

# 清洗重复数据 
data = {
    'name':['Iven', 'Iven', 'Rosenn', 'Starry'],
    'age':[21, 21, 22, 23] # 数据重读
}
df = pd.DataFrame(data)
# duplicated() 如果重复返回True 否则False 
print(df.duplicated()) 
# drop_duplicates()删除重复数据
df.drop_duplicates(inplace=True)
print(df)