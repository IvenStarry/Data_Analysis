import numpy as np
# bool_ int_ uint8 float_ complex_

# todo dtype(object, align, copy) 转换为的数据类型对象 True填充字段使其类似C的结构体 复制dtype对象，若为false则是对内置数据类型对象的引用
dt = np.dtype(np.int32)
print(dt)
# int8 int16 int32 int64 四种数据类型可以使用字符串'i1' 'i2' 'i4' 'i8'代替
dt = np.dtype('i4')
print(dt)
# 字节顺序标注  <意味着小端法（低位字节存储在低位地址） >意味着大端法（高位字节存放在低位地址）
dt = np.dtype('<i4')
print(dt)
# 创建结构化数据类型
dt = np.dtype([('age', np.int8)])
print(dt)
# 数据类型应用于ndarray对象
a = np.array([(10, ), (20, ), (30, )], dtype=dt)
print(a)
# 类型字段名用于存取实际的age列
print(a['age'])
# 定义一个结构化数据类型，将这个dtype应用于ndarray对象
student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
print(student)
a = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student)
print(a)

"""
内建类型符号
b bool
i int
u uint
f float
c 复数浮点型
m timedelta 时间间隔
M datetime 日期时间
O python对象
S, a (byte)字符串
U unicode 统一码 编码格式 数字到文字的映射
V 原始数据(void)
"""