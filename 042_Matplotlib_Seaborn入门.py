'''
Seaborn 是一个建立在matplotlib基础之上的python数据可视化库
简化统计数据可视化过程，提供高级接口和美观的默认主题
提供高级接口，轻松绘制散点图、折线图、柱状图、热图等
注重美观性，绘图更吸引人
'''
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
'''
sns.set_theme() 选择不同的主题和模板

主题theme: 
darkgrid(默认)：深色网格主题
whitegold：浅色网格主题
dark：深色主题 没有网络
white：浅色主题 没有网络
ticks：深色主题 带有刻度标记

模板Context
paper：适用于小图，具有较小的标签和线条
notebook(默认)：适用于笔记本电脑和类型环境，具有中等大小的标签和线条
talk：适用于演讲幻灯片，大尺寸标签和线条
poster： 适用于海报，非常大的标签和线条
'''
sns.set_theme(style='whitegrid', palette='pastel')
products = ['product A', 'product B', 'product C', 'product D']
sales = [120, 210, 150, 180]
sns.barplot(x=products, y=sales) # 创建柱状图
plt.xlabel('products')
plt.ylabel('sales')
plt.title('product sales by category')
plt.show()

# * 绘图函数
# 散点图 sns.scatterplot() 用于绘制两个变量之间的散点图，可选择增加趋势线
data = {'A': [1, 2, 3, 4, 5], 'B':[5, 4, 3, 2, 1]}
df = pd.DataFrame(data)
sns.scatterplot(x='A', y='B', data=df)
plt.show()

# 折线图 sns.lineplot() 绘制变量随着另一个变量变化的趋势线图
data = {'X': [1, 2, 3, 4, 5], 'Y':[5, 4, 3, 2, 1]}
df = pd.DataFrame(data)
sns.lineplot(x='X', y='Y', data=df)
plt.show()

# 柱状图 sns.barplot() 绘制变量的均值或其他聚合函数的柱状图
data = {'Category':['A', 'B', 'C'], 'Value':[3, 7, 5]}
df = pd.DataFrame(data)
sns.barplot(x='Category', y='Value', data=df)
plt.show()

# 箱线图 sns.boxplot() 绘制变量的分布情况，包括中位数、四分位数等
data = {'Category':['A', 'A', 'B', 'B', 'C', 'C'], 'Value':[3, 7, 5, 9, 2, 6]}
df = pd.DataFrame(data)
sns.boxplot(x='Category', y='Value', data=df)
plt.show()

# 热图 sns.heatmap() 绘制矩阵数据的热图，展示相关性矩阵
data = {'A': [1, 2, 3, 4, 5], 'B':[5, 4, 3, 2, 1]}
df = pd.DataFrame(data)
correlation_matrix = df.corr() # 创建一个相关性矩阵
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# 小提琴图 sns.violinplot() 显示分布的形状和密度估计，结合了箱线图和核密度估计
data = {'Category':['A', 'A', 'B', 'B', 'C', 'C'], 'Value':[3, 7, 5, 9, 2, 6]}
df = pd.DataFrame(data)
sns.violinplot(x='Category', y='Value', data=df)
plt.show()