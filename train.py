import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
from VGG19 import Network,scriptPath
import numpy as np
from utils.training import loadCfgFile,saveCfgFile,write_summary
from utils.log import getCurrentTime,getTimeLeft,printInPlace
from utils.metrics import accuracy
from time import time,sleep
import numpy as np
import cv2
from shutil import copyfile,copytree,rmtree
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

def flipCoin():
    return np.random.randint(0,2)

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

        if flipCoin:
            img = tf.image.flip_left_right(img)
        if flipCoin:
            img += tf.random_uniform(input_shape,maxval = 0.15)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.7, 1.3)
        
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

def scriptsBackup(cfg):
    ############## copy all scripts to the output directoy specified in the cfg file
    modelPath = scriptPath()
    cwd = os.path.dirname(os.path.realpath(__file__))
    copyfile(modelPath,cfg['output_dir'] + os.sep + os.path.basename(modelPath))
    copyfile(cfgPath,cfg['output_dir'] + os.sep + 'cfg.json')
    copyfile(os.path.realpath(__file__),cfg['output_dir'] + os.sep + 'train.py')
    copyfile(cwd + os.sep + 'test.py',cfg['output_dir'] + os.sep + 'test.py')
    if not os.path.isdir(cfg['output_dir'] + os.sep + 'utils'):
        copytree(cwd + os.sep + 'utils',cfg['output_dir'] + os.sep + 'utils')
    else:
        rmtree(cfg['output_dir'] + os.sep + 'utils')
        copytree(cwd + os.sep + 'utils',cfg['output_dir'] + os.sep + 'utils')


def model_and_optimizer_initialization(model,optimizer,cfg):
    dummyData = tf.zeros([ cfg['batch_size'] ] + cfg['input_shape'])
    #### compute a dummy training pass for initializing all variables in model and optimizer objects. This is required for later variable restoring
    dummyLabels = np.argmax(np.random.random([ cfg['batch_size'] ] + cfg['output_shape']),axis=1)
    dummyLabels = tf.one_hot(dummyLabels,depth=2)
    prev_lr = optimizer._lr
    optimizer._lr = 0 ### disable learning for dummy data
    with tf.GradientTape() as tape:
        preds = model(dummyData,training=True)
        batch_loss = tf.losses.softmax_cross_entropy(dummyLabels,preds)
        reg_loss = tf.reduce_mean(model.losses)
        total_loss = batch_loss + reg_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                            global_step=tf.train.get_or_create_global_step())
    
    optimizer._lr = prev_lr ### reset optimizer learning rate   

############## ENABLE EAGER EXECUTION
tf.enable_eager_execution()

################################ LOAD TRAIN CONFIG FILE
cwd = os.path.dirname(os.path.realpath(__file__))
cfgPath = cwd + os.sep + 'cfg.json'
cfg = loadCfgFile(cfgPath)
if not os.path.isdir(cfg['output_dir']):
    os.makedirs(cfg['output_dir'])

bestModelDir = os.path.join(cfg['output_dir'], 'bestModel')
lastModelDir = os.path.join(cfg['output_dir'], 'lastModel')

############## COPY SCRIPTS USED TO THE OUTPUT DIRECTORY
scriptsBackup(cfg)

############################## DATASETS LOADING AND MODEL INITIALIZATION
train_dataset = load_dataset(cfg['train_tfrecord'],input_shape = cfg['input_shape'],batch_size = cfg['batch_size'],shuffle_buffer_size = 10000)
val_dataset = load_dataset(cfg['val_tfrecord'],input_shape = cfg['input_shape'],batch_size = cfg['batch_size'],shuffle_buffer_size = 0)

model = Network()
optimizer = tf.train.AdamOptimizer()
##### build and initialize the model and optimizer by feedforwarding some dummy train batch
model_and_optimizer_initialization(model,optimizer,cfg)

save_var_list = model.variables + optimizer.variables() + [tf.train.get_global_step()] #### save model variables, optimizer state and global training step
saver = tf.contrib.eager.Saver(var_list = save_var_list)

############################## CREATE TENSORBOARD SUMMARY FILE
currTime = getCurrentTime()
summariesDir = os.path.join(cfg['output_dir'], 'tensorboard', currTime)
summary_writer = tf.contrib.summary.create_file_writer(summariesDir,flush_millis = 10000)
print('\n\nTensorboard command:\ntensorboard --port 3468 --logdir='+summariesDir)

############################### TRAINING OPERATION
with open(cfg['train_tfrecord']+'_info.csv','r') as f:
    num_train_samples = len(f.readlines())

with open(cfg['val_tfrecord']+'_info.csv','r') as f:
    num_val_samples = len(f.readlines())

train_steps_per_epoch = num_train_samples // cfg['batch_size']
val_steps_per_epoch = num_val_samples // cfg['batch_size']


if cfg['train_mode'] == 'start':
    init_epoch = 0
    best_val_loss = np.inf
    best_val_acc = 0
    train_step = 0
    print('\nSTARTING TRAINING...')

else:
    init_epoch = cfg['last_epoch'] + 1
    if init_epoch >= cfg['train_epochs']:
        raise Exception('\nInitial training epoch value is higher than the max. number of training epochs specified at the cfg file')
    best_val_loss = cfg['best_val_loss']
    best_val_acc = cfg['best_val_acc']
    if os.path.isdir(lastModelDir):
        saver.restore(os.path.join(lastModelDir,'lastModel'))
        print('\nRESUMING TRAINING FROM CKPT at: %s & EPOCH: %d ...' % (lastModelDir,init_epoch))
    else:
        raise Exception('\nError: RESUME MODE SELECTED BUT NO PREVIOUS LAST MODEL WAS AVAILABLE')

for epoch in range(init_epoch,cfg['train_epochs']):
    curr_lr = lr_schedule(cfg['init_learning_rate'],epoch=epoch)
    optimizer._lr = curr_lr
    print('\n\nLearning rate: %f' % optimizer._lr)

    ####### TRAIN LOOP 
    ep_train_loss_sum = 0
    ep_train_acc_sum = 0
    startTime = time()
    for n,train_batch in enumerate(train_dataset):
        imgs,genders = train_batch

        with tf.GradientTape() as tape:
            logits = model(imgs,training=True)
            batch_loss = tf.losses.softmax_cross_entropy(genders,logits)
            reg_loss = tf.reduce_mean(model.losses)
            total_loss = batch_loss + reg_loss

        if np.isnan(batch_loss) or np.isinf(batch_loss):  ###### Terminate training on NaN or Inf loss value
           raise Exception('\nInvalid loss value encountered. Stopping training!!')
        
        batch_acc = accuracy(genders,logits)

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                            global_step=tf.train.get_or_create_global_step())
            
        ep_train_loss_sum += batch_loss
        ep_train_acc_sum += batch_acc
        ep_mean_train_loss = (ep_train_loss_sum) / (n + 1)
        ep_mean_train_acc = (ep_train_acc_sum) / (n + 1)
        time_left = getTimeLeft(startTime,n,train_steps_per_epoch)

        write_summary(summary_writer,batch_loss,summaryName = 'train_loss',summaryType = 'scalar')
        write_summary(summary_writer,batch_acc,summaryName = 'train_acc',summaryType = 'scalar')
            
        printInPlace('Epoch: %d (training) -- Batch: %d/%d -- ETA: %s -- Mean loss: %f -- Mean acc: %f -- Batch reg loss: %f'
                     % (epoch,n,train_steps_per_epoch,time_left,ep_mean_train_loss,ep_mean_train_acc,reg_loss))
    
    write_summary(summary_writer,ep_mean_train_loss,summaryName = 'train_mean_loss',summaryType = 'scalar',step = epoch)
    write_summary(summary_writer,ep_mean_train_acc,summaryName = 'train_mean_acc',summaryType = 'scalar',step = epoch)

    print(' ')

    ######## VALIDATION LOOP
    ep_val_loss_sum = 0
    ep_val_acc_sum = 0
    startTime = time()
    for n,val_batch in enumerate(val_dataset):
        imgs,genders = val_batch
        logits = model(imgs,training=False)
        batch_loss = tf.losses.softmax_cross_entropy(genders,logits)

        batch_acc = accuracy(genders,logits)

        ep_val_loss_sum += batch_loss
        ep_val_acc_sum += batch_acc
        ep_mean_val_loss = (ep_val_loss_sum) / (n + 1)
        ep_mean_val_acc = (ep_val_acc_sum) / (n + 1)
        time_left = getTimeLeft(startTime,n,val_steps_per_epoch)

        printInPlace('Epoch: %d (validation) -- Batch: %d/%d -- ETA: %s -- Mean loss: %f -- Mean acc: %f'
                     % (epoch,n,val_steps_per_epoch,time_left,ep_mean_val_loss,ep_mean_val_acc))
    
    write_summary(summary_writer,ep_mean_val_loss,summaryName = 'val_mean_loss',summaryType = 'scalar',step = epoch)
    write_summary(summary_writer,ep_mean_val_acc,summaryName = 'val_mean_acc',summaryType = 'scalar',step = epoch)
          
    if ep_mean_val_loss < best_val_loss:
        ######## BEST MODEL CHECKPOINT WRITTING
        saver.save(file_prefix=os.path.join(bestModelDir,'bestModel'))
        
        print('\n\tMean validation loss decreased from %f to %f. \n\tModel saved to: %s' % (best_val_loss,ep_mean_val_loss,bestModelDir))
        best_val_loss = ep_mean_val_loss
    
    ######## LAST MODEL CHECKPOINT WRITTING
    saver.save(file_prefix=os.path.join(lastModelDir,'lastModel'))

    cfg['last_epoch'] = epoch
    cfg['best_val_loss'] = float(best_val_loss)
    cfg['best_val_acc'] = float(best_val_acc)
    saveCfgFile(cfg,cfgPath) #### update original cfg file
    saveCfgFile(cfg,cfg['output_dir'] + os.sep + 'cfg.json') ##### update backup cfg file in output folder
