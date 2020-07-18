# encoding=utf8
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random

#conv-BN-RELU
def conv_module(x, filters,kernel=3, use_bn = True, use_act = True,stride=1, downsampling = False, padding='same'):
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=(kernel,kernel),
                               strides=(stride,stride),padding=padding,use_bias=False)(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
    if use_act:
        x = tf.keras.layers.ReLU()(x)

    if downsampling:
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(x)
    return x

#上采样模块
def upsample_module(x, filters, kernel=3):
    x = tf.keras.layers.Conv2DTranspose(filters=filters,kernel_size=(kernel,kernel),strides=(2,2),
                                        padding='same',use_bias=False,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.02))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

#预测objectness模块
def head_objectness(x, num_class, filters=64):
    hm = conv_module(x,filters,kernel=1,stride=1,use_act=False, use_bn=False)
    # 预测层不用bn/act
    hm = conv_module(hm,num_class,kernel=1,stride=1,use_act=False, use_bn=False)
    return hm
#预测height-width的模块
def head_hw(x, num_class, filters=64):
    wh = conv_module(x, filters, kernel=1, stride=1,use_act=False, use_bn=False)
    #预测层不用bn/act
    wh = conv_module(wh, num_class * 2, kernel=1, stride=1, use_act=False, use_bn=False)
    return wh

#最终的模型
def get_detnet(input_shape, num_class):
    input = tf.keras.layers.Input(shape=input_shape)
    x = conv_module(input,16,downsampling=True)
    x = conv_module(x, 32, downsampling=True)
    x = conv_module(x, 64, downsampling=True)
    x = conv_module(x, 128, downsampling=True)
    x = conv_module(x, 256, downsampling=True)
    x = upsample_module(x,128)
    x = upsample_module(x,64)
    x = upsample_module(x,32)

    #两个输入模块
    objectness_heatmap = head_objectness(x,num_class)
    hw_heatmap = head_hw(x, num_class)

    #使用tf.keras.Model()把输入和输出组合在一起
    return tf.keras.Model(input,[objectness_heatmap,hw_heatmap],name="detnet")


def TEST_DETECTION():
    num_class = 1
    input_shape = (96,224,3)
    net = get_detnet(input_shape,num_class)
    x = np.random.uniform(-1,1,size=(4,96,224,3))
    x = tf.convert_to_tensor(x)
    objectness, hw = net(x)
    print(net.summary())
    print(x.shape, objectness.shape, hw.shape)


if __name__ == "__main__":
    TEST_DETECTION()


