'''
CSV文件：逗号分隔值文件
np.savetxt(frame, array, fmt='%.18e', delimiter=None)
frame:文件、字符串或产生器 可以是.gz .bz2的压缩文件
array:存入文件的数组
fmt:存入文件的格式 %d %.2f %.18e
delimiter:分割字符串。默认为任何空格
'''
# 整数保存
import numpy as np
a = np.arange(100).reshape(5, 20)
np.savetxt('related_data/test1.csv', a, fmt='%d', delimiter=',')
# 浮点数保存
a = np.arange(100).reshape(5, 20)
np.savetxt('related_data/test2.csv', a, fmt='%.1f', delimiter=',')

'''
np.loadtxt(frame, dtype=np.float, delimiter=None, unpack=False)
frame:文件、字符串或产生器 可以是.gz .bz2的压缩文件
dtype:数据类型可选
delimiter:分割字符串。默认为任何空格
unpack:若为True，读入属性将分别写入不同变量
'''
# 默认浮点型
b = np.loadtxt('related_data/test1.csv', delimiter=',')
print(b)
# 指定整数型
# * 新版numpy无int类型 使用int_
b = np.loadtxt('related_data/test1.csv', dtype=np.int_, delimiter=',')
print(b)

# csv只能有效存储一维和二维数组即load/save函数只能存取一维和二维数组