import pandas as pd
import numpy as np

# todo 创建DataFrame
'''
pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)

data：DataFrame 的数据部分，可以是字典、二维数组、Series、DataFrame 或其他可转换为 DataFrame 的对象。如果不提供此参数，则创建一个空的 DataFrame。
index：DataFrame 的行索引，用于标识每行数据。可以是列表、数组、索引对象等。如果不提供此参数，则创建一个默认的整数索引。
columns：DataFrame 的列索引，用于标识每列数据。可以是列表、数组、索引对象等。如果不提供此参数，则创建一个默认的整数索引。
dtype：指定 DataFrame 的数据类型。可以是 NumPy 的数据类型，例如 np.int64、np.float64 等。如果不提供此参数，则根据数据自动推断数据类型。
copy：是否复制数据。默认为 False，表示不复制数据。如果设置为 True，则复制输入的数据。
'''

# 使用列表创建
data = [['Iven', 21], ['Rosenn', 25], ['Starry', 19]]
df = pd.DataFrame(data, columns=['Name', 'Age'])
df['Name'] = df['Name'].astype(str) # astype方法设置每列的数据类型
df['Age'] = df['Age'].astype(float)
print(df)

# 使用字典创建
data = {'Name':['Iven', 'Rosenn', 'Starry'], 'Age':[21, 25, 19]}
df = pd.DataFrame(data)
print(df)

# 使用ndarray对象创建 长度必须相同，若传递了index，索引长度要等于数组长度，若未传递索引，默认是range(n),n是数组长度
ndarray_data = np.array([['Iven', 21], ['Rosenn', 25], ['Starry', 19]])
df = pd.DataFrame(ndarray_data, columns=['Name', 'Age'])
print(df)

# 使用字典创建 key为列名
data = [{'a':1, 'b':2}, {'a':5, 'b':10, 'c':20}]
df = pd.DataFrame(data)
print(df) # 没有相对应的数据为NAN

# 使用Series创建
s1 = pd.Series(['Iven', 'Rosenn', 'Starry'])
s2 = pd.Series([21, 23, 24])
s3 = pd.Series(['Chengdu', 'Hangzhou', 'Xi\'an'])
df = pd.DataFrame({'Name':s1, 'Age':s2, 'City':s3})
print(df)

# todo 访问DataFrame元素
# 访问行 loc 返回指定行的数据，若没有设置索引，第一行索引为0，第二行为1
data = {'calories': [420, 280, 400], 'duration':[50, 45, 40]}
df = pd.DataFrame(data)
print(df.loc[0])
print(df.loc[1]) # 返回结果是一个Series数据
print(df.loc[[0,1]]) # 返回一个dataframe数据

# 访问列 loc[] iloc[]
data = {'calories': [420, 280, 400], 'duration':[50, 45, 40]}
df = pd.DataFrame(data, index=['day1', 'day2', 'day3']) # 指定索引值
print(df)
print(df.loc['day2']) # df.loc[row_label, column_label] 索引名 day1 day2
print(df.iloc[:, 0])  # df.iloc[row_index, column_index] 整数 0, 1, ...

# 访问单个元素 [列][行] 先得到一列Series 在得到Series中的一个数据
print(df['calories'][0]) 

# todo DataFrame 属性和方法
print(df.shape)      # 形状
print(df.columns)    # 列名
print(df.index)      # 索引
print(df.head())     # 前几行数据，默认前五行
print(df.tail())     # 后几行数据，默认后五行
print(df.info())     # 数据信息
print(df.describe()) # 描述统计信息
print(df.mean())     # 求平均值
print(df.sum())      # 求和
print(df.max())      # 最大值
print(df.min())      # 最小值

# todo 修改DataFrame
# 修改列数据
df['calories'] = [200, 300, 400]
# 修改行数据
df.loc['day1'] = [100, 90]
# 添加新列
df['food'] = ['rice', 'banana', 'apple']
# 添加新行 loc 指定特定索引添加新行 concat 合并两个或多个DataFrame  append(已被弃用)
df.loc['day4'] = [500, 60, 'bread']
new_row = pd.DataFrame([[600, 70, 'cococola']], index=['day5'], columns=['calories', 'duration', 'food'])
df = pd.concat([df, new_row], ignore_index=False)
print(df)
# // new_row = {'calories':600, 'duration':70, 'food':'cococola'}
# // df = df.append(new_row, ignore_index=True)
# 删除列 drop
df = df.drop('duration', axis=1)  # 删除轴1的duration，即列名为duration的列
print(df)
# 删除行 drop
df_drop = df.drop('day1') # 默认删除轴0的day1，即行名为day1的行
print(df_drop)

# todo 索引操作
# 重置索引 reset_index
df_reset = df.reset_index(drop=True)
print(df_reset)
# 设置索引 set_index 将一列设置为索引 drop是否保留被转换的列
df_set = df.set_index(['calories'], drop=True)
print(df_set)
# 布尔索引
print(df[df['calories'] > 400])
'''
append(idx)         连接另一个index对象，产生新的index对象
diff(idx)           计算差集，产生新的index对象
intersection(idx)   计算交集
union(idx)          计算并集
delete(loc)         删除loc位置的元素
insert(loc,e)       在loc位置添加一个元素e  
'''
df_del = df.columns.delete(1)
print(df_del)
df_ins = df.index.insert(1, 'day6')
print(df_ins)
print(df)
df_rei = df.reindex(index=df_ins, columns=df_del)
print(df_rei)
df_rei = df.reindex(index=df_ins, columns=df_del, method='bfill')
print(df_rei)
df_rei = df.reindex(index=df_ins, columns=df_del, method='ffill')
print(df_rei)

# 重排索引 
'''
reindex(index=None, columns=None, ...) 重拍已有的序列

index,columns 新的行列自定义索引
fill_value    重新索引中，用于填充缺失位置的值
method        填充方法，ffill 会将上一个非 NaN 的值填充到此位置  bfill会将下一个非 NaN 的值填充到此位置 
                跟行列插入的位置无关(1,)(不是找索引0 索引1的数值)  跟索引值名称有关(找day6前一个索引day5数值和后一个索引NAN)
limit         最大填充量
copy          True生成新对象
'''
df_reindex = df.reindex(index = ['day1', 'day3', 'day2', 'day5', 'day4'], columns=['food', 'calories'])
print(df)
print(df_reindex)
new_df_insert = df_reindex.columns.insert(2, '新增') # insert 在列的指定位置加新列后的所有列名
print(new_df_insert)
new_df = df_reindex.reindex(columns=new_df_insert, fill_value=200)
print(new_df)

# todo 数据类型
# 查看数据类型 dtypes
print(df.dtypes)
# 转换数据类型 astype
df['calories'] = df['calories'].astype('float32')
print(df)

# todo 合并与分割
# concat
'''
pd.concat([df1, df2], ignore_index=True, join='outer',axis=1)

axis：0轴（默认）按行拼接（增加行）, 1轴，按列拼接（增加列）；
join：outer默认（拼接时取并集）；inner（拼接时取交集）
ignore_index：默认False，即不重置dataframe的索引；True重置索引，从0开始

'''
new_row = pd.DataFrame([[700, 'fenta']], index=['day6'], columns=['calories', 'food'])
df_row_concat = pd.concat([df, new_row])
print(df_row_concat)
new_col = pd.DataFrame({'name':['Iven', 'Rosenn', 'Starry', 'Bob', 'Alen']}, index=['day1', 'day2', 'day3', 'day4', 'day5'])
df_col_concat = pd.concat([df, new_col], axis=1)
print(df_col_concat)

# merge
'''
pd.merge(df1,df2,on='列名',how='outer')   
how：合并方式：how = 'inner'（默认）类似于取交集  'outer'，类似于取并集 left以左表为主表 right以右表为主表
on： 用于连接的列名，若不指定则以两个Dataframe的列名的交集作为连接键
'''
new_col = pd.DataFrame({'food':['rice', 'banana', 'apple', 'bread', 'cococola'],  'name':['Iven', 'Rosenn', 'Starry', 'Bob', 'Alen']})
df_col_merge = pd.merge(df, new_col, how='outer')
print(df_col_merge)
new_row = pd.DataFrame({'calories':900, 'food':'orange'},index=['day6'])
df_row_merge = pd.merge(df, new_row, how='outer')
print(df_row_merge)


# todo 算术运算
# 根据行列索引，补齐后运算，默认产生浮点数，补齐时填充NAN。二维和一维、一维和零维间为广播运算，+ - * / 运算产生新对象
a = pd.DataFrame(np.arange(12).reshape(3, 4))
b = pd.DataFrame(np.arange(20).reshape(4, 5))
print(a)
print(b)
print(a + b)
print(a * b)
'''
add(d, **argws)  类型间加法运算
sub(d, **argws)  类型间减法运算
mul(d, **argws)  类型间乘法运算
div(d, **argws)  类型间除法运算
'''
print(b.add(a, fill_value=8888)) # 哪空替代哪 在运算
print(a.mul(b, fill_value=1))

# 不同维度间为广播运算，一维Series默认在轴1运算
a = pd.Series(np.arange(4))
b = pd.DataFrame(np.arange(20).reshape(4, 5))
print(a - 10)
print(b - a) # b每行减去a的转置
print(b.sub(a, axis=0)) # 使用运算方法可以让一维Series在轴0运算

# todo 比较运算法则
# 只能比较相同索引的元素，不可以补齐，二维和一维、一维和零维间为广播运算，> < >= <= == !=运算产生布尔对象
a = pd.DataFrame(np.arange(12).reshape(3, 4))
b = pd.DataFrame(np.arange(12, 0, -1).reshape(3, 4))
print(a)
print(b)
print(a > b)
print(a == b)

# 不同维度广播
a = pd.Series(np.arange(4))
b = pd.DataFrame(np.arange(12).reshape(3, 4))
print(a)
print(b)
print(a > b)
print(b > 0)