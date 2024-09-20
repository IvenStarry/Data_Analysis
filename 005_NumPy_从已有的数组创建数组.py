import numpy as np

# asarray(a, dtype, order) 类似array 参数比array少俩
x = [1, 2, 3]
a = np.asarray(x)
print(a)

# 元组转换ndarray
x = (1, 2, 3)
a = np.asarray(x)
print(a)

# 元组列表转换ndarray 高版本无法生成非同质数组
# // x = [(1, 2 , 3), (4, 5)]
# // a = np.asarray(x)
# // print(a)

# frombuffer(buffer, dtype, count=-1, offset) 实现动态数组 接受buffer输入参数 以流的形式读取转化成ndarray对象 
# offset读取起始位置 默认0   b" "前缀表示：后面字符串是bytes 类型
s = b'Hello World!'
a = np.frombuffer(s, dtype='S1')
print(a)

# fromiter(iterable, dtype, count=-1) 从迭代对象中建立ndarray对象，返回一维数组
list = range(5)
it = iter(list)
x = np.fromiter(it, dtype=float)
print(x)