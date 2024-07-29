import matplotlib.pyplot as plt
import numpy as np

'''
imsave(fname, arr, **kwargs) 将图像数据保存在磁盘上 保存在指定目录 支持PNG JPEG BMP等多种图像格式
kwargs: 可选。用于指定保存的图像格式以及图像质量等参数
'''

img_data = np.random.random((100, 100)) # 功能同rand 但random传入一个完整的元组 rand接收分开的参数
plt.imshow(img_data)
plt.imsave('related_data/test.png', img_data)

# 创建灰度图像
img_gray = np.random.random((100, 100))
# 创建彩色图像
img_color = np.zeros((100, 100, 3))
img_color[:, :, 0] = np.random.random((100, 100))
img_color[:, :, 1] = np.random.random((100, 100))
img_color[:, :, 2] = np.random.random((100, 100))

plt.imshow(img_gray, cmap='gray')
plt.imsave('related_data/test_gray.png', img_gray, cmap='gray')
plt.imshow(img_color)
plt.imsave('related_data/test_color.jpg', img_color) # 若未指定图像格式 默认保存JPEG格式