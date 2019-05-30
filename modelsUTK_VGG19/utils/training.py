import numpy as np
import json
import tensorflow as tf

def loadCfgFile(cfgPath):
    with open(cfgPath,'r') as fp:
        cfgDict = json.load(fp)
    return cfgDict

def saveCfgFile(cfgDict,cfgPath):
    with open(cfgPath,'w') as fp:
        json.dump(cfgDict,fp,indent=4)

def batchSplit(dataList,batchSize):
    #### splits a data list in batches of size = batchSize
    lData = len(dataList)
    dataBatches = np.array_split(dataList,np.arange(batchSize,lData,batchSize))
    return dataBatches

def write_summary(summary_writer,tensor,summaryName,summaryType,step = None):
    if step is None:
        step = tf.train.get_global_step()
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        if summaryType == 'scalar':
            tf.contrib.summary.scalar(summaryName,tensor,step=step)
        elif summaryType == 'image':
            tf.contrib.summary.image(summaryName,tensor,step=step)




