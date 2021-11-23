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
from utils import load_Cifar10

class VGG_():
  def __init__(self):
    self.model = tf.keras.Sequential(name='VGG_type')
    self.x_shape = [32,32,3]

  def VGG_block(self, weight_decay, width, p = 2):
  
    self.model.add(Conv2D(width, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
    self.model.add(BatchNormalization())
    self.model.add(Conv2D(width, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
    self.model.add(BatchNormalization())
    self.model.add(MaxPooling2D(pool_size=(p,p)))

  def VGG_init(self, dropout_rate, weight_decay, width, block_numbers = 3):
    Dropout_coef = dropout_rate

    self.model.add(Conv2D(width, (3, 3), padding='same', input_shape=self.x_shape, activation='relu'))
    self.model.add(BatchNormalization())
    self.model.add(Conv2D(width, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
    self.model.add(BatchNormalization())
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    
    for i in range(block_numbers):
      self.VGG_block(weight_decay, width, p = 2)

    self.VGG_block(weight_decay, width, p = 1)

    self.model.add(Flatten())
    self.model.add(Dense(512, activation='relu'))
    self.model.add(Dropout(Dropout_coef))
    self.model.add(Dense(512, activation='relu'))
    self.model.add(Dropout(Dropout_coef))
    self.model.add(Dense(10))
    self.model.add(Activation('softmax'))
    return self.model

def VGG():
  VGG_class = VGG_()
  deep_model = VGG_class.VGG_init(0.2, 0.001, 128)
  deep_model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['acc'])
  return deep_model
