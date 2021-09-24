import LoadData as ld
import random
from scipy.io import FortranFile
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


###############################################################################################
######################################### Train ###############################################


# batch, nz, nx, input_maps
x = tf.placeholder(tf.float32, [None, None, None, 3])
# batch, output_maps
y = tf.placeholder(tf.float32, [None, None])
LearnRate = tf.placeholder(tf.float32, shape=[])
#beta = tf.placeholder(tf.float32, shape=[])
phase_train = tf.placeholder(tf.bool, name='phase_train')



# learning iteration per same learning rate
subEpoch = 100
nStep = 5
totalEpoch = subEpoch * nStep
totalBatch = 1

# initial learning rate & lr decays to lr*1/5 per N_epoch
iniLR = 0.0005


for i1 in range(nStep):
    # Change Learningrate
    rate = iniLR * np.power(5.0, -float(i1))

    for i2 in range(subEpoch):

        for i3 in range(totalBatch):

            x_data, y_data = ld.GetBatch()
            sess = tf.Session()
            optimizer = 'adam'
            outPut = sess.run(optimizer,
                         feed_dict={x: x_data, y: y_data, LearnRate: rate, phase_train: True})
            print(outPut)
###############################################################################################
###############################################################################################
