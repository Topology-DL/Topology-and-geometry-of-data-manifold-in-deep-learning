import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add 
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from keras import backend as K
from keras import regularizers

relu6 = tf.keras.layers.ReLU(6.)

def _conv_block(inputs, filters, kernel, strides):
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return relu6(x)

def _bottleneck(inputs, filters, kernel, t, s, r=False):
    tchannel = inputs.shape[-1] * t
    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))
    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = relu6(x)
    x = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if r:
        x = tf.keras.layers.add([x, inputs])
    return x

def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    x = _bottleneck(inputs, filters, kernel, t, strides)
    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)
    return x

def MobileNetV2_init(input_shape, k):
   
    inputs = tf.keras.layers.Input(shape=input_shape, name='input')
    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))
    width = 128
    x = _inverted_residual_block(x, width, (3, 3), t=1, strides=2, n=3)
    x = _inverted_residual_block(x, width, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, width, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, width, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, width, (3, 3), t=6, strides=2, n=3)

    x = _conv_block(x,  width, (1, 1), strides=(1, 1))
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1,  width))(x)
    x = tf.keras.layers.Dropout(0.3, name='Dropout')(x)
    x = tf.keras.layers.Conv2D(k, (1, 1), padding='same')(x)
    x = tf.keras.layers.Activation('softmax', name='final_activation')(x)
    output = tf.keras.layers.Reshape((k,), name='output')(x)
    model = tf.keras.models.Model(inputs, output)
    return model

def MobileNetV2():
  deep_model = MobileNetV2_init((32, 32, 3), 10)
  deep_model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['acc'])
  return deep_model
