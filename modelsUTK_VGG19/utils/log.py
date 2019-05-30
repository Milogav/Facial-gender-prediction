from datetime import datetime
from time import time,strftime

def printInPlace(text):
    print('\r'+text+'\t'*5,end='',sep = '')

def getTimeLeft(startTime,currStep,totalSteps):
    ##### assumed first step is step = 0
    elaspsedTime = time() - startTime
    estTimePerStep = elaspsedTime / (currStep + 1)
    remainingSteps = totalSteps - currStep
    leftTime = estTimePerStep * remainingSteps #### in seconds
    M,S = divmod(leftTime,60)
    H,M = divmod(M,60)
    return "%d:%02d:%02d" % (H, M, S) 

def log(logfile,string,printStr = True):
    logfile.write(string+'\n')
    logfile.flush()
    if printStr:
        print(string)
    
def getCurrentTime():
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

