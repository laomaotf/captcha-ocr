# encoding=utf8
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
################################################
#参考 https://github.com/FLming/CRNN.tf2



def vgg_style(input_tensor):
    """
    The original feature extraction structure from CRNN paper.
    Related paper: https://ieeexplore.ieee.org/abstract/document/7801919
    """
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(512, 2, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def get_crnn(input_shape, num_classes):
    img_input = keras.Input(shape=input_shape)
    x = vgg_style(img_input)
    # assert x.shape = (batch_size, 1, N, 512)
    # N is time steps used in following RNN module
    x = layers.Reshape((-1, 512))(x)

    x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x)
    x = layers.Dense(units=num_classes)(x) #logist output
    return keras.Model(inputs=img_input, outputs=x, name='CRNN')


def TEST_CRNN():
    num_class = 10
    input_shape = (32,100,3)
    net = get_crnn(input_shape,num_class)
    x = np.random.uniform(-1,1,size=(4,32,100,3))
    x = tf.convert_to_tensor(x)
    output = net(x)
    print(net.summary())
    print(x.shape, output.shape)


if __name__ == "__main__":
    TEST_CRNN()




