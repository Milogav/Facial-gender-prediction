import tensorflow as tf
import numpy as np

def euclidean_distance(vects,eps = 1e-08):
    x,y = vects
    d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), 1)+eps)
    return d

def cosine_distance(vects):
    return 1 - euclidean_distance(vects) / 2

def abs_diff(vects):
    x, y = vects
    return tf.abs(x - y)

def siamese_accuracy(y_true,logits,thr):
    preds = tf.cast(logits < thr, y_true.dtype)
    matches = tf.cast(tf.equal(y_true,preds),tf.float32)
    return tf.reduce_mean(matches)

def accuracy(one_hot_labels,logits):
    preds = tf.math.argmax(tf.nn.softmax(logits),1)
    labels = tf.math.argmax(one_hot_labels,1)
    matches = tf.cast(tf.equal(preds,labels),tf.float32)
    acc = tf.math.reduce_mean(matches)
    return acc

