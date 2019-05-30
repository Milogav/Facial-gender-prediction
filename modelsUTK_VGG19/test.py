import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
from VGG19 import Network,scriptPath
import numpy as np
from utils.training import loadCfgFile
from utils.log import getCurrentTime,getTimeLeft,printInPlace
from time import time,sleep
import numpy as np
import cv2
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

def load_dataset(tfr_path,input_shape,batch_size,shuffle_buffer_size,multithreading = True):
    
    def _parse_function(proto):
        keys_to_features = {'img' : tf.FixedLenFeature([], tf.string),
                'gender' : tf.FixedLenFeature([], tf.int64)}
        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)
        # Turn your saved image string into an array (decoding)
        img = tf.image.decode_jpeg(parsed_features['img'],channels = input_shape[-1])

        gender = parsed_features["gender"]
        ###### input preprocessing
        img = tf.cast(img,tf.float32) / 127.5 - 1
        img = tf.image.resize_images(img,input_shape[0:2])
        
        gender = tf.one_hot(gender,2)
        return img,gender

    dataset = tf.data.TFRecordDataset(tfr_path)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    if multithreading:
        dataset = dataset.map(_parse_function, num_parallel_calls=cpu_count())
    else:
        dataset = dataset.map(_parse_function)
    
    # Set the number of datapoints you want to load and shuffle
    if shuffle_buffer_size: 
        dataset = dataset.shuffle(shuffle_buffer_size)
    
    # Set the batchsize
    dataset = dataset.batch(batch_size)

    return dataset

def lr_schedule(init_lr,epoch):
    decay_factor = 0.95
    return init_lr * decay_factor**epoch

############## ENABLE EAGER EXECUTION
tf.enable_eager_execution()

################################ LOAD TRAIN CONFIG FILE
cwd = os.path.dirname(os.path.realpath(__file__))
cfgPath = cwd + os.sep + 'cfg.json'
cfg = loadCfgFile(cfgPath)

############################## DATASETS LOADING AND MODEL INITIALIZATION
test_dataset = load_dataset(cfg['test_tfrecord'],input_shape = cfg['input_shape'],batch_size = cfg['batch_size'],shuffle_buffer_size = 0)
print('Loaded test set: '+cfg['test_tfrecord'])
bestModelDir = os.path.join(cfg['output_dir'], 'bestModel')

model = Network()
dummyData = tf.zeros([ cfg['batch_size'] ] + cfg['input_shape'])
model(dummyData,training=False)  ##### initialize model parameters by feedforwarding some data
del dummyData

saver = tf.contrib.eager.Saver(var_list = model.variables)
saver.restore(os.path.join(bestModelDir,'bestModel'))
##print('\nLOADING CKPT: %s...' % bestModelSaver.latest_checkpoint)

with open(cfg['test_tfrecord']+'_info.csv','r') as f:
    num_test_samples = len(f.readlines())
    
test_steps_per_epoch = num_test_samples // cfg['batch_size']
test_loss_sum = 0
pred_genders = list()
truth_genders = list()

startTime = time()
for n,test_batch in enumerate(test_dataset):
    imgs,genders = test_batch
    logits = model(imgs,training=False)
    preds = tf.argmax(tf.nn.softmax(logits),axis=1)
    truth = tf.argmax(genders,axis=1)
    pred_genders.append(preds.numpy())
    truth_genders.append(truth.numpy())
                 
    batch_loss = tf.losses.softmax_cross_entropy(genders,logits)
    test_loss_sum += batch_loss
    mean_test_loss = (test_loss_sum) / (n + 1)
    time_left = getTimeLeft(startTime,n,test_steps_per_epoch)

    printInPlace('Test -- Batch: %d/%d -- ETA: %s -- Mean loss: %f'
                     % (n,test_steps_per_epoch,time_left,mean_test_loss))
    
truth_genders = np.concatenate(truth_genders)
pred_genders = np.concatenate(pred_genders) 

acc = accuracy_score(truth_genders,pred_genders)
cm = confusion_matrix(truth_genders,pred_genders)

print('\n\nTEST ACCURACY: ',acc)
print('\nTEST CONFUSION MATRIX:')
print(cm)


    
