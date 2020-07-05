# coding=utf-8
import os
import numpy as np
from tqdm import tqdm
from PIL import Image


# python读取mnist数据
num_data = 60000
fd = open('train-images-idx3-ubyte','rb')
loaded = np.fromfile(file=fd, dtype=np.uint8)
trainX = loaded[16:].reshape((num_data, 28, 28, 1)).astype(np.int64)


fd = open('train-labels-idx1-ubyte','rb')
loaded = np.fromfile(file=fd, dtype=np.uint8)
trainY = loaded[8:].reshape((num_data)).astype(np.int64)

os.makedirs("../sample/charset/",exist_ok=True)
for img, label, i in tqdm(zip(trainX, trainY, range(num_data))):
    img = np.reshape(img, [28,28]).astype(np.uint8)
    outpath = os.path.join('../sample/charset/{}'.format(label))
    os.makedirs(outpath,exist_ok=True)
    outpath = os.path.join(outpath,"mnist_%d.jpg"%i)
    img = Image.fromarray(img)
    img.save(outpath)

