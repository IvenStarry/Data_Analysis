import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
hist(x, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid',
    orientation='vertical', rwidth=None, log=False, color=None, stacked=False, **kwargs)

x：表示要绘制直方图的数据，可以是一个一维数组或列表。
bins：可选参数，表示直方图的箱数。默认为10。
range：可选参数，表示直方图的值域范围，可以是一个二元组或列表。默认为None，即使用数据中的最小值和最大值。
density：可选参数，表示是否将直方图归一化。默认为False，即直方图的高度为每个箱子内的样本数，而不是频率或概率密度。
weights：可选参数，表示每个数据点的权重。默认为None。
cumulative：可选参数，表示是否绘制累积分布图。默认为False。
bottom：可选参数，表示直方图的起始高度。默认为None。
histtype：可选参数，表示直方图的类型，可以是'bar'、'barstacked'、'step'、'stepfilled'等。默认为'bar'。
align：可选参数，表示直方图箱子的对齐方式，可以是'left'、'mid'、'right'。默认为'mid'。
orientation：可选参数，表示直方图的方向，可以是'vertical'、'horizontal'。默认为'vertical'。
rwidth：可选参数，表示每个箱子的宽度。默认为None。
log：可选参数，表示是否在y轴上使用对数刻度。默认为False。
color：可选参数，表示直方图的颜色。
label：可选参数，表示直方图的标签。
stacked：可选参数，表示是否堆叠不同的直方图。默认为False。
**kwargs：可选参数，表示其他绘图参数。
'''
data = np.random.randn(1000)
plt.hist(data, bins=30, color='skyblue', alpha=0.8)
plt.title('hist test')
plt.xlabel('value')
plt.ylabel('frequency')
plt.show()

data1 = np.random.normal(0, 1, 100000) # 正态分布随机数组 mu sigma size
data2 = np.random.normal(2, 1, 100000)
data3 = np.random.normal(-2, 1, 100000)

plt.hist(data1, bins=30, alpha=0.5, label='Data 1')
plt.hist(data2, bins=30, alpha=0.5, label='Data 2')
plt.hist(data3, bins=30, alpha=0.5, label='Data 3')

plt.title('hist test')
plt.xlabel('value')
plt.ylabel('frequency')
plt.legend() # 显示图例
plt.show()

# * 结合pandas
random_data = np.random.normal(170, 10, 250)
dataframe = pd.DataFrame(random_data) # 数据转换为DataFrame
dataframe.hist() # 使用Pandas.hist() 方法绘制直方图
plt.title('Pandas hist test')
plt.xlabel('X-value')
plt.ylabel('y-value')
plt.show()

# 使用series对象绘制直方图 将数据框的列替换为series对象即可
data = pd.Series(np.random.normal(size=100))
plt.hist(data, bins=10)
plt.title('Pandas hist test')
plt.xlabel('X-value')
plt.ylabel('y-value')
plt.show()