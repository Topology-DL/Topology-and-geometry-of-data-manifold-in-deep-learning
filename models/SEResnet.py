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

def senet_block(inputs, ratio):
    shape = inputs.shape
    channel_out = shape[-1]
    squeeze = layers.GlobalAveragePooling2D()(inputs)
    shape_result = layers.Flatten()(squeeze)
    shape_result = layers.Dense(int(channel_out / ratio), activation='relu')(shape_result)

    shape_result = layers.Dense(channel_out, activation='sigmoid')(shape_result)

    excitation_output = tf.reshape(shape_result, [-1, 1, 1, channel_out])

    h_output = excitation_output * inputs
    return h_output

def res_block(input, input_filter, output_filter):
    res_x = layers.Conv2D(filters=output_filter, kernel_size=(3, 3), activation='relu', padding='same')(input)
    res_x = layers.BatchNormalization()(res_x )
    res_x = senet_block(res_x, 8)
    res_x = layers.Conv2D(filters=output_filter, kernel_size=(3, 3), activation=None, padding='same')(res_x )
    res_x = layers.BatchNormalization()(res_x )
    res_x = senet_block(res_x, 8)
    if input_filter == output_filter:
        identity = input
    else: 
        identity = layers.Conv2D(filters=output_filter, kernel_size=(1,1), padding='same')(input)

    x = layers.Add()([identity, res_x])
    output = layers.Activation('relu')(x)
    return output

def get_SE_Resnet():
    inputs = tf.keras.Input(shape=(32,32,3), name='img')
    h1 = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
    h1 = layers.BatchNormalization()(h1)
    h1 = senet_block(h1, 8)

    block1 = res_block(h1, 64, 128)
    block1 = layers.MaxPool2D(pool_size=(2, 2))(block1)

    block2 = res_block(block1, 128, 128)
    block2 = layers.MaxPool2D(pool_size=(2, 2))(block2)

    block3 = res_block(block2, 128, 128)
    block3 = layers.MaxPool2D(pool_size=(2, 2))(block3)

    block4 = res_block(block3, 128, 128)
    block4 = layers.MaxPool2D(pool_size=(2, 2))(block4)

    block5 = res_block(block4, 128, 128)
    block5 = layers.MaxPool2D(pool_size=(2, 2))(block5)

    block6 = res_block(block5, 128, 128)
    block6 = layers.MaxPool2D(pool_size=(1, 1))(block6)

    out = layers.Flatten()(block6)
    out = layers.Dense(10)(out)
    outputs = layers.Activation('softmax')(out)

    deep_model = tf.keras.Model(inputs, outputs, name='se-resnet')
    optimizer = tf.keras.optimizers.Adamax(lr=0.001)
    deep_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return deep_model

def SEResnet():
  deep_model = get_SE_Resnet()
  return deep_model
