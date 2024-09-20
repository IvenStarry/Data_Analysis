from PIL import Image
import numpy as np

im = np.array(Image.open("related_data/dog.jpg"))
# 图像是一个三维数组，维度分别是高度、宽度和像素RGB值
print(im.shape, im.dtype)