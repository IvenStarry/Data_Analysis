import numpy as np

# 位运算
a1 = np.array([True, False, True], dtype=bool)
a2 = np.array([False, True, False], dtype=bool)

result_and = np.bitwise_and(a1, a2) # 按位与
result_or = np.bitwise_or(a1, a2) # 按位或
result_xor = np.bitwise_xor(a1, a2) # 按位异或
result_not = np.bitwise_not(a1) # 按位取反

print('and:', result_and)
print('or:', result_or)
print('xor:', result_xor)
print('not', result_not)

# invert 按位取反 1 00000001 2 00000010    补码取反 -1 11111110 转原码 10000010  补码取反 11111101 转原码 10000011
print('Invert:', np.invert(np.array([1, 2], dtype=np.int8)))

# left_shift 左移位运算
print(bin(5),bin(np.left_shift(5, 2)))
print('Left shift:', np.left_shift(5, 2)) # 00101 -> 10100

# right_shift 右移位运算
print(bin(10),bin(np.left_shift(10, 1)))
print('Right shift', np.right_shift(10, 1)) # 01010 -> 00101

'''
操作符运算
& 与运算
| 或运算
^ 异或运算
~ 取反运算
<< 左移运算
>> 右移运算
''' 