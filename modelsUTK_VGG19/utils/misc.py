import numpy as np
from imgproc import normalization

def derange(array,maxTry=1000):
    # shuffles a iterable ensuring that none of the elements
    # remains at its original position
    c = 0
    while True:
        c += 1
        d_array = np.random.permutation(array)
        if all(array != d_array):
            break
        elif c > maxTry:
                print('Maximum number of dearangement attempts reached ('+str(maxTry)+'). Aborting...')
                break

    return d_array

def imshow(img,winName = 'image'):
    disp = normalization(img,min_val=0,max_val=255).astype(np.uint8)
    cv2.imshow('winName',disp)
    cv2.waitKey()
    cv2.destroyWindow(winName)