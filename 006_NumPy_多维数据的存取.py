'''
a.tofile(frame, sep='', format='%s')
frame: 文件、字符串
sep: 数据分割字符串 如果为空串 写入文件为二进制
format: 写入数据格式
'''
import numpy as np

a= np.arange(100).reshape(5, 10, 2)
# 只是逐项输出数据 看不出维度信息
a.tofile("related_data/test1.dat", sep=',', format='%d')
# 存储为二进制文件 占用空间更小 如果可以知道显示字符的编码以及字节之间的关系就可以实现转化
a.tofile("related_data/test2.dat", format='%d')

'''
np.fromfile(frame, dtype=float, count=-1, sep='')
frame:文件、字符串
dtype:读取的数据类型
count:读入元素个数,-1表示读入全部文件
sep: 数据分割字符串 如果为空串 写入文件为二进制

读取时需要知道存入文件时数组的维度和元素类型
可以通过再写一个元数组文件存储维度和元素类型信息
'''
# 文本文件
b = np.fromfile('related_data/test1.dat', dtype=np.int_, sep=',')
print(b)
b = np.fromfile('related_data/test1.dat', dtype=np.int_, sep=',').reshape(5, 10, 2)
print(b)

# 二进制文件 无需指定分隔符
c = np.fromfile("related_data/test2.dat", dtype=np.int_).reshape(5, 10 ,2)
print(c)

'''
NumPy便捷文件存取 固定文件格式
正常文件：np.save(fname, array)   压缩：np.savez(fname, array)
fname: 文件名 以.npy为扩展名 压缩扩展名为.npz
array: 数组变量

np.load(fname)
fname: 文件名 以.npy为扩展名 压缩扩展名为.npz

可以直接还原数组维度和元素类型信息
因为在文件开头用显示的方式 将数组的源信息存储在了第一行
'''
np.save("related_data/test1.npy", a)
d = np.load("related_data/test1.npy")