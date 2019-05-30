import tensorflow as tf
import numpy as np

def contrastive_loss(Y_true, distances,margin):
    return tf.reduce_mean(Y_true*tf.square(distances)+(1 - Y_true)*tf.maximum((margin-distances),0))

def triplet_loss(y_true,dAP,dAN,margin):
    #### dAP = distance vector between the embbedings of the anchor and positive images
    #### dAN = distance vector between the embbedings of the anchor and negative images
    diff = dAP - dAN
    return tf.reduce_mean(tf.maximum((diff + margin),0))
