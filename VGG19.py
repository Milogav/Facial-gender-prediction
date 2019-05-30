import tensorflow as tf
import os
import numpy as np

def scriptPath():
    return os.path.realpath(__file__)

class Network(tf.keras.Model):  #### USING SUBCLASSING API
     def __init__(self):
          super(Network, self).__init__()
          vgg19 = tf.keras.applications.VGG19(weights = 'imagenet',include_top = False)
          self.vgg = tf.keras.Model(vgg19.input,vgg19.output)
          for layer in self.vgg.layers[:-15]:
               layer.trainable = False
          self.flat = tf.keras.layers.Flatten()
          self.dense1 = tf.keras.layers.Dense(units = 1024,activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(0.001))
          self.drop1 = tf.keras.layers.Dropout(rate = 0.5)
          self.dense2 = tf.keras.layers.Dense(units = 1024,activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(0.001))
          self.drop2 = tf.keras.layers.Dropout(rate = 0.5)
          self.dense3 = tf.keras.layers.Dense(units = 2,kernel_regularizer=tf.keras.regularizers.l2(0.001))


     def call(self, inputs,training):
          x = self.vgg(inputs)
          x = self.flat(x)
          x = self.dense1(x)
          x = self.drop1(x,training = training)
          x = self.dense2(x)
          x = self.drop2(x,training = training)
          x = self.dense3(x)

          return x


