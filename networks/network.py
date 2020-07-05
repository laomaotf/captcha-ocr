import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random



class CBRP(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(out_channels,3,activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(out_channels, 3, activation="relu")
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.pooling = tf.keras.layers.MaxPool2D(2)
        return


    def call(self,data):
        x1 = self.bn1(self.conv1(data))
        x2 = self.bn2(self.conv2(data))
        return self.pooling(x1 + x2)

class CNN(tf.keras.Model):
    def __init__(self, num_classes = 1000):
        super().__init__()
        self.stage1 = CBRP(16)
        self.stage2 = CBRP(32)
        self.stage3 = CBRP(64)
        self.stage4 = CBRP(128)
        self.stage5 = CBRP(256)
        #self.flatten = tf.keras.layers.GlobalMaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512,activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(num_classes)
        return


    def call(self,data):
        x = self.stage1(data)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        return x


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    """Upsamples an input.
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_dropout: If True, adds the dropout layer
    Returns:
      Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    #elif norm_type.lower() == 'instancenorm':
    #    result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())
    return result

def UNET_DOWN(input_shape):
    #base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
    #base_model = tf.keras.applications.MobileNetV2(input_shape = input_shape,include_top=False,weights="imagenet")
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None)

    # 使用这些层的激活设置
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # 创建特征提取模型
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    #down_stack.trainable = False  # 不可训练
    return down_stack

def UNET(input_shape,num_classes=1000):
    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),  # 32x32 -> 64x64
    ]
    # 这是模型的最后一层
    last = tf.keras.layers.Conv2DTranspose(
        num_classes, 3, strides=2,
        padding='same', activation='softmax')  # 64x64 -> 128x128

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    down_stack = UNET_DOWN(input_shape=input_shape)
    # 在模型中降频取样
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])
    # 升频取样然后建立跳跃连接
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)



def TEST_CNN():
    image = tf.io.read_file('../readme_image/iCaptcha.jpg')
    image = tf.image.decode_image(image,channels=3)
    image = tf.image.convert_image_dtype(image,tf.float32)
    new_shape = (128,128)
    image = tf.image.resize(image,new_shape)
    image = image[tf.newaxis,:]

    net = CNN(10)
    output = net(tf.constant(image))
    print(net.summary())
    print('input shape: ',image.shape)
    print('output shape: ',output.shape)
    print(output.numpy())

def TEST_UNET():
    image = tf.io.read_file('../readme_image/iCaptcha.jpg')
    image = tf.image.decode_image(image,channels=3)
    image = tf.image.convert_image_dtype(image,tf.float32)
    new_shape = [224,224]
    image = tf.image.resize(image,new_shape)
    image = image[tf.newaxis,:]

    net = UNET(input_shape=new_shape + [3],num_classes=10)
    output = net(tf.constant(image))
    print(net.summary())
    print('input shape: ',image.shape)
    print('output shape: ',output.shape)
    #print(output.numpy())

if __name__ == "__main__":
    TEST_UNET()

