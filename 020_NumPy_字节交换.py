import numpy as np

# np.adarray.byteswap() 将ndarray每个元素的字节进行大小端转换
a = np.array([1, 245, 8755], dtype=np.int16)
print(a)
print(list(map(hex, a))) # hex 16进制编码
print(a.byteswap(inplace=True)) # 传入True原地交换
print(list(map(hex, a))) 
'''
大端模式
1：0000 0000 0000 0001 
245：0000 0000 1111 0101  
8755：0010 0010 0011 0011

小端模式
1：0000 0001 0000 0000 
245：(首1补码)1111 0101 0000 0000 (转原码表示)1000 1011 0000 0000  
8755：0011 0011 0010 0010
'''