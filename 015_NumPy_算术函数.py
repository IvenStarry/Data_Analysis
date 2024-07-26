import numpy as np

# 加减乘除 add subtract multiply divide
a = np.arange(9, dtype=np.float64).reshape(3, 3)
b = np.array([10, 10, 10])
print(np.add(a, b))
print(np.subtract(a, b))
print(np.multiply(a, b))
print(np.divide(a, b))

# reciprocal() 返回参数各元素的倒数
a = np.array([0.25, 1.33, 1, 100])
print(a)
print(np.reciprocal(a))

# power() 将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂 x^n
a = np.array([10, 100, 1000])
print(a)
print(np.power(a, 2))
b = np.array([1, 2, 3])
print(np.power(a, b))

# np.mod np.remainder 计算输入数组中相应元素的相除后的余数 
a = np.array([10, 20, 30])
b = np.array([3, 5, 7])
print(np.mod(a, b))
print(np.remainder(a, b))


a = np.arange(24).reshape((2, 3, 4))
print(f"平方运算:{np.square(a)}")
print(f"开方运算：{np.sqrt(a)}")
print(f"整数小数分离{np.modf(np.sqrt(a))}") # np.modf()将整数和小数部分分成两个部分

b = np.sqrt(a)
print(a)
print(b)
print(np.maximum(a, b)) # 输出结果浮点数
print(a > b)