import LoadData as ld
import random
from scipy.io import FortranFile
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


###############################################################################################
######################################### Train ###############################################

# learning iteration per same learning rate
subEpoch = 100
nStep = 5
totalEpoch = subEpoch * nStep
totalBatch = 1

# initial learning rate & lr decays to lr*1/5 per N_epoch
iniLR = 0.0005


def GetBatch():
    nTrainField = 100
    TrainData = ld.LoadData(nTrainField, 3000, 4)
    return TrainData, np.zeros([1, 1], dtype=float)


for i1 in range(nStep):
    # Change Learningrate
    rate = iniLR * np.power(5.0, -float(i1))

    for i2 in range(subEpoch):

        for i3 in range(totalBatch):

            x_data, y_data = GetBatch()
            sess = tf.Session()
            optimizer='adam'
            _ = sess.run(optimizer,
                         feed_dict={x: x_data, y: y_data, LearnRate: rate, phase_train: True})

###############################################################################################
###############################################################################################
