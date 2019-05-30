import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import matplotlib.pyplot as plt
import tensorflow as tf
from VGG19 import Network,scriptPath
import numpy as np
from utils.training import loadCfgFile,saveCfgFile,write_summary
from utils.log import getCurrentTime,getTimeLeft,printInPlace
from utils.metrics import euclidean_distance
from utils.losses import contrastive_loss
from time import time,sleep
import numpy as np
import cv2
from shutil import copyfile,copytree,rmtree
from multiprocessing import cpu_count

############## ENABLE EAGER EXECUTION
tf.enable_eager_execution()

################################ LOAD TRAIN CONFIG FILE
cwd = os.path.dirname(os.path.realpath(__file__))
cfgPath = cwd + os.sep + 'cfg.json'
cfg = loadCfgFile(cfgPath)

bestModelDir = os.path.join(cfg['output_dir'], 'bestModel')

############################## DATASETS LOADING AND MODEL INITIALIZATION
##train_dataset = load_dataset(cfg['train_tfrecord'],input_shape = cfg['input_shape'],batch_size = cfg['batch_size'],shuffle_buffer_size = 10000)

model = Network()
dummyData = tf.zeros([ cfg['batch_size'] ] + cfg['input_shape'])
model(dummyData,training=False)  ##### initialize model parameters by feedforwarding some data
del dummyData

saver = tf.contrib.eager.Saver(var_list = model.variables)
saver.restore(os.path.join(bestModelDir,'bestModel'))

vc = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cwd + '/haarcascade_frontalface_default.xml')
genders = ['Female','Male']
while True:
    ret, frame = vc.read()
    img = frame
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces):
        x,y,w,h = faces[0]
        img = img.astype(np.float32) / 127.5 - 1
        img = cv2.resize(img[y:y+h, x:x+w],(cfg['input_shape'][0],cfg['input_shape'][1]),interpolation=cv2.INTER_LINEAR) 
        
        pred = model(np.expand_dims(img,axis=0),training=False)
        pred = tf.nn.softmax(pred).numpy()
        genderPred = genders[np.argmax(pred)]

        textLabel = '%s - %.3f' % (genderPred,np.max(pred))
        cv2.putText(frame,textLabel,(x,y), cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,thickness=2, color=(255,0,0))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('Gender prediction', frame)
    if cv2.waitKey(20) >= 0:
        break

cv2.destroyAllWindows()




#     for n,val_batch in enumerate(val_dataset):
#         imgs,ages = val_batch
#         preds = model(imgs,training=False)
#         batch_loss = tf.losses.softmax_cross_entropy(ages,preds)

#         ep_val_loss_sum += batch_loss
#         ep_mean_val_loss = (ep_val_loss_sum) / (n + 1)
#         time_left = getTimeLeft(startTime,n,val_steps_per_epoch)

#         printInPlace('Epoch: %d (validation) -- Batch: %d/%d -- ETA: %s -- Mean loss: %f'
#                      % (epoch,n,val_steps_per_epoch,time_left,ep_mean_val_loss))
    
#     write_summary(summary_writer,ep_mean_val_loss,summaryName = 'val_mean_loss',summaryType = 'scalar',step = epoch)
          
#     if ep_mean_val_loss < best_val_loss:
#         ######## BEST MODEL CHECKPOINT WRITTING
#         bestModelSaver.save()
#         print('\n\tMean validation loss decreased from %f to %f. \n\tModel saved to: %s' % (best_val_loss,ep_mean_val_loss,bestModelDir))
#         best_val_loss = ep_mean_val_loss
    
#     ######## LAST MODEL CHECKPOINT WRITTING
#     lastModelSaver.save()

#     cfg['last_epoch'] = epoch
#     cfg['best_val_loss'] = float(best_val_loss)
#     saveCfgFile(cfg,cfgPath) #### update original cfg file
#     saveCfgFile(cfg,cfg['output_dir'] + os.sep + 'cfg.json') ##### update backup cfg file in output folder
