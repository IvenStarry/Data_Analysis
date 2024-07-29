import matplotlib.pyplot as plt
import numpy as np

'''
imread(fname, format=None) 从图像文件中读取图像数据，返回一个np.ndarray对象。形状(nrows, ncols, nchannels) 灰度图像通道数1 彩色图像通道数3或4 红绿蓝还有alpha通道
format: 指定图像文件格式，若不指定，默认根据文件后缀自动识别格式
'''

img = plt.imread('related_data/map.jpeg')
plt.imshow(img)
plt.show()

# 更改numpy数组修改图像 这里使图像变暗
img_map = img / 255
plt.figure(figsize=(10, 6))

for i in range(1, 5):
    plt.subplot(2, 2, i)
    x = 1 - 0.2 *(i - 1)
    plt.axis('off')
    plt.title('x={:.1f}'.format(x))
    plt.imshow(img_map * x)
plt.show()