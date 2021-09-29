import numpy as np
import tensorflow as tf
import time
from scipy.io import FortranFile
import random
import os

######################################################################################################
########################################## Loaddata ##################################################

##Data Parameters
#grid size of data field
nxp = 192
nzp = 192


def LoadData(nField, iniField, intervalField):
    #r.m.s of data (0 : du/dy, 1 : dw/dy, 2 : p)
    Input_std = np.empty([3])
    Input_std[0] = 66.274826292
    Input_std[1] = 37.0097954241
    Input_std[2] = 1.58318422533
    # [number of field, z-grids, x-grids, variables(du/dy, dw/dy, p, dT/dy)]
    Data = np.empty([nField, nzp, nxp, 4], dtype=np.float32)
    for j in range(nField):
        p = iniField + j*intervalField
        # folder = os.path.dirname(os.path.abspath(__file__))
        # Filename = 'DLdata\\' + '%05d' % (p)
        # Filename = os.path.join(folder, Filename) 
        Filename = 'DLdata/' + '%05d' % (p)

        f = FortranFile(Filename, 'r')
        insfield = f.read_reals(dtype='float32')
        Data[j, :, :, :] = np.transpose(
            np.reshape(insfield, (4, nxp, nzp), order='F'))

    for image1 in range(3):
        Data[:, :, :, image1] = Data[:, :, :, image1]/Input_std[image1]
    return Data

def GetBatch():
    nTrainField = 100
    TrainData = LoadData(nTrainField, 3000, 4)
    TrainData = TrainData[:,:,:, 0:3]
    #TrainData = tf.slice(TrainData, begin=[0, 0, 0, 0], size=[100, 192, 192, 3])
    #TrainData = np.array(t2)
    #return tf.convert_to_tensor(TrainData), tf.convert_to_tensor(np.zeros([1, 1], dtype=float))
    #return tf.convert_to_tensor(TrainData), tf.convert_to_tensor(np.zeros([1, 1], dtype=float))
    #return TrainData,tf.placeholder(tf.float32, [100, 3])
    return TrainData,np.zeros([100, 3], dtype=np.float32)

# nTrainField = 100
# TrainData = LoadData(nTrainField, 3000, 4)
# print("sucess")
