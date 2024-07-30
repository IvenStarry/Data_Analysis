import pandas as pd
import json
from glom import glom

'''
JSON Java Script Objection Notation  Java Script对象表示法，是存储和交换文本信息的语法，类似XML
JSON比XML更小、更快、更易解析
pandas可以处理JSON数据
'''

# 读取JSON文件 read_json
df = pd.read_json('related_data/test.json')
print(df.to_string())

# 也可以直接处理JSON字符串
data = [
    {
        "id": "1",
        "name": "Iven",
        "age": 19
    },
    {
        "id": "2",
        "name": "Rosenn",
        "age": 21
    },
    {
        "id": "3",
        "name": "Bob",
        "age": 22
    }
]
df = pd.DataFrame(data)
print(df)

# 字典转DataFrame JSON对象与字典具有相同的格式
s = {
    'col1':{'row1':1, 'row2':2, 'row3':3},
    'col2':{'row4':4, 'row5':5, 'row6':6},
}
df = pd.DataFrame(s) # 读取JSON文件转dataframe
print(df)

# 内嵌的JSON数据文件
df = pd.read_json('related_data/nested_list.json')
print(df)
# 这时候需要json_normalize()将内嵌的数据完整解析 record_path设置要展开的内嵌数据名 meta展示无需展开的其他数据名
with open('related_data/nested_list.json', 'r') as f: 
    data = json.loads(f.read()) # 使用python的json模块读取数据 字典结构
df_nested_list = pd.json_normalize(data, record_path=['students'], meta=['school_name', 'class']) 
print(df_nested_list)

# 更复杂的json文件
with open ('related_data/nested_mix.json', 'r') as f:
    data = json.loads(f.read())
df = pd.json_normalize(data, record_path=['students'], meta=['class', ['info', 'president'], ['info', 'contacts', 'tel']])
print(df)

# 读取内嵌数据中的一组数据 glom模块允许我们使用. 来访问内嵌对象的属性
df = pd.read_json('related_data/nested_deep.json')
data = df['students'].apply(lambda row: glom(row, 'grade.math'))
print(data)