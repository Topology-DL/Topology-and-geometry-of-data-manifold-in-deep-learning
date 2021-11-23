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
import numpy as np

# resnet code from "https://github.com/suvoooo/Learn-TensorFlow"
regular = 0.001
initializer = tf.keras.initializers.HeNormal()

def res_identity(x, filters): 
  x_skip = x 
  f1, f2 = filters
  activation = tf.keras.activations.elu

  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(regular),kernel_initializer=initializer)(x)
  x = BatchNormalization()(x)
  x = Activation(activation)(x)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(regular),kernel_initializer=initializer)(x)
  x = BatchNormalization()(x)
  x = Activation(activation)(x)
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(regular),kernel_initializer=initializer)(x)
  x = BatchNormalization()(x)
  x = Add()([x, x_skip])
  x = Activation(activation)(x)

  return x

def res_conv(x, s, filters):
  activation = tf.keras.activations.elu
  x_skip = x
  f1, f2 = filters

  x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(regular), kernel_initializer=initializer)(x)
  x = BatchNormalization()(x)
  x = Activation(activation)(x)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(regular),kernel_initializer=initializer)(x)
  x = BatchNormalization()(x)
  x = Activation(activation)(x)
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(regular),kernel_initializer=initializer)(x)
  x = BatchNormalization()(x)
  x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(regular),kernel_initializer=initializer)(x_skip)
  x_skip = BatchNormalization()(x_skip)

  x = Add()([x, x_skip])
  x = Activation(activation)(x)
  return x

def Resnet():
  width = 128
  activation = tf.keras.activations.elu
  input_im = Input(shape=(32,32,3)) 
  x = ZeroPadding2D(padding=(3, 3))(input_im)

  x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = BatchNormalization()(x)
  x = Activation(activation)(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)
  x = res_conv(x, s=1, filters=(width, width))
  x = res_identity(x, filters=(width, width))
  x = res_conv(x, s=2, filters=(width, width))
  x = res_identity(x, filters=(width, width))
  x = res_conv(x, s=2, filters=(width, width)) 
  x = res_identity(x, filters=(width, width))
  x = res_conv(x, s=1, filters=(width, width))
  x = res_identity(x, filters=(width, width))
  x = res_conv(x, s=2, filters=(width, width))
  x = res_identity(x, filters=(width, width))

  x = layers.GlobalAveragePooling2D()(x)
  x = Flatten()(x)
  x = tf.keras.layers.Dense(10)(x)
  x = tf.keras.layers.Activation('softmax')(x)

  deep_model = Model(inputs=input_im, outputs=x, name='Resnet50')
  deep_model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['acc'])
  return deep_model

