import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''
df.cov()  计算协方差矩阵
df.corr() 计算相关系数矩阵 pearson spearman kendall系数

在pandas中，数据的相关性分析是通过计算不同变量之间的相关系数来了解它们之间的关系
数据相关性是一项重要的分析任务，可以帮助我们理解各个变量之间的关系
使用 corr() 方法计算数据集中每列之间的关系

df.corr(method='pearson', minperiods=1)
method (可选): 字符串类型，用于指定计算相关系数的方法。默认是 'pearson'，
                还可以选择 'kendall'（Kendall Tau 相关系数）或 'spearman'（Spearman 秩相关系数）。
min_periods (可选): 表示计算相关系数时所需的最小观测值数量。默认值是 1，即只要有至少一个非空值，就会进行计算。
                    如果指定了 min_periods，并且在某些列中的非空值数量小于该值，则相应列的相关系数将被设为 NaN。

返回一个相关系数矩阵，矩阵的行和列对应数据框的列名，矩阵的元素是对应列之间的相关系数。

常见的相关性系数包括 Pearson 相关系数和 Spearman 秩相关系数：
Pearson 相关系数: 即皮尔逊相关系数，用于衡量了两个变量之间的线性关系强度和方向。它的取值范围在 -1 到 1 之间，其中 -1 表示完全负相关，
                    1 表示完全正相关，0 表示无线性相关。可以使用 corr() 方法计算数据框中各列之间的 Pearson 相关系数。
Spearman 相关系数：即斯皮尔曼相关系数，是一种秩相关系数。用于衡量两个变量之间的单调关系，即不一定是线性关系。
                它通过比较变量的秩次来计算相关性。可以使用 corr(method='spearman') 方法计算数据框中各列之间的 Spearman 相关系数。
'''

data = {'A':[1, 2, 3, 4, 5], 'B':[5, 4, 3, 2, 1]}
df =pd.DataFrame(data)
# 计算 pearson 相关系数
correlation_matrix = df.corr()
print(correlation_matrix) # 因为数据集是线性相关的，因此主对角线值为1，副对角线值为-1完全负相关
# 计算 spearman 秩相关系数
correlation_matrix = df.corr(method='spearman')
print(correlation_matrix) # 结果与pearman相关系数矩阵相同，因为两个变量之间完全负相关

# 可视化相关性   使用Seaborn库绘制更加精美的相关系数矩阵图像
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f') # 绘制热力图 annot: 默认为False，为True的话，会在格子上显示数字
plt.show()