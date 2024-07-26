import numpy as np

# sin cos  tan arccos arcsin arctan
a = np.array([0, 30, 45, 60, 90])
print(np.sin(a * np.pi / 180))
print(np.cos(a * np.pi / 180))
print(np.tan(a * np.pi / 180))
# np.degrees(将弧度转换为角度)
print(np.degrees(np.arcsin(np.sin(a * np.pi / 180)))) 
print(np.degrees(np.arccos(np.cos(a * np.pi / 180))))
print(np.degrees(np.arctan(np.tan(a * np.pi / 180))))

# around(a, decimals) 返回四舍五入值 decimals:舍入的位数 默认0  如果为负数，整数将四舍五入到小数点左侧的位置
a = np.array([1.0, 5.55, 123, 0.567, 25.532])
print(a)
print(np.around(a))
print(np.around(a, decimals=1))
print(np.around(a, decimals=-1))
print(np.around(a, decimals=-2))

# floor 返回小于或等于表达式的最大整数 向下取整
# ceil  返回大于或等于表达式的最小整数 向上取整
a = np.array([-1.7, 1.5, -0.2, 0.6, 10])
print(np.floor(a))
print(np.ceil(a))