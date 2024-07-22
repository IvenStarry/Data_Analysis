import numpy as np
a = np.arange(24).reshape((2, 3, 4))
print(a)

# 数组与标量之间的运算作用于数组的每一个元素
a.mean()
a = a / a.mean()
print(a)

a = np.arange(24).reshape((2, 3, 4))
print(f"平方运算:{np.square(a)}")
print(f"开方运算：{np.sqrt(a)}")
print(f"整数小数分离{np.modf(np.sqrt(a))}") # np.modf()将整数和小数部分分成两个部分

b = np.sqrt(a)
print(a)
print(b)
print(np.maximum(a, b)) # 输出结果浮点数
print(a > b)