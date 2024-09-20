import numpy as np

# char.add() 两个数组的字符串连接
print(np.char.add('hello', 'world'))
print(np.char.add(['hello', 'hi'], ['world', 'nihao~']))

# char.multiply(a, num) 执行多重连接 num重复次数
print(np.char.multiply('Iven', 5))

# char.center(str, width, fillchar) 将字符串居中，指定字符在左侧和右侧进行填充 width:填充后整体长度 fillchar:填充字符
print(np.char.center('Iven', 10, fillchar='*'))

# char.capitalize() 将字符串的第一个字母转换大写
print(np.char.capitalize('rosennn'))

# char.title() 对数组的每个单词的第一个字母转为大写
print(np.char.title('rosenn enjoys surfing'))

# char.lower() 对数组的每个元素转换小写，对每个元素调用str.lower
print(np.char.lower('IVEN'))

# char.upper() 对数组的每个元素转换大写，对每个元素调用str.upper
print(np.char.upper(['iven', 'rosenn']))

# char.split(str, sep) 指定分隔符对字符串进行分割，返回数组 默认分隔符是空格
print(np.char.split('i like coding'))
print(np.char.split('www.github.com'), sep='.')

# char.splitlines() 以换行符作为分隔符来分割字符串，返回数组  \r\n都可以作为换行符
print(np.char.splitlines('Iven\nlikes it'))
print(np.char.splitlines('Iven\rlikes it'))

# char.strip() 移除开头或结尾的特定字符
print(np.char.strip('abbbbacc', 'a'))

# char.join() 通过指定分隔符来连接数组中的元素或字符串
print(np.char.join([':', '-'], ['Iven', 'Starry']))

# char.replace(str, old, new) 使用新字符串替换字符串的所有子字符串
print(np.char.replace('i like coffee', 'ff', 'fffff'))

# char.encode() 对数组中每个元素都调用str.encode函数进行编码,默认编码UTF-8
print(np.char.encode('Iven','cp500')) # cp500是编码类型
print(np.char.encode('Iven','ascii')) # ascii是编码类型
print(np.char.encode('Iven'))         # 默认是uft-8编码

# char.decode() 对编码元素进行str.decode()解码
# char.encode() 对数组中每个元素都调用str.encode函数,默认编码UTF-8
a = np.char.encode('Iven','cp500')
b = np.char.encode('Iven','ascii')
c = np.char.encode('Iven')
print(np.char.decode(a,'cp500')) # cp500是编码类型
print(np.char.decode(b,'ascii')) # ascii是编码类型
print(np.char.decode(c))         # 默认是uft-8编码
