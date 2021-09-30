import random
import sys
import LoadData as ld
from scipy.io import FortranFile
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

########################### Model Parameters ######################################################
# input kernel size (e.g. 33x33)
nxl = 16
nxr = 16
nzl = 16
nzr = 16
nx = nxl + 1 + nxr
nz = nzr + 1 + nzl

#####################################################################################################
######################################## DEEP LEARNING MODEL ########################################


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv_layer(x, W, b, padding):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding) + b
    # return tf.nn.relu(conv)
    return tf.nn.elu(conv)


def convlayer_bn(x, W, padding, phase_train):
    moving_mean = 0.9
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
    conv_bn = tf.layers.batch_normalization(
        conv, momentum=moving_mean, training=phase_train)  # , name='conv1_bn')
    return conv_bn


def fc_layer(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID') + b
    return conv


SizeO = tf.placeholder(tf.int32, shape=[])
# batch, nz, nx, input_maps
x = tf.placeholder(tf.float32, [None, None, None, 3])
# batch, output_maps
y = tf.placeholder(tf.float32, [None, None])
LearnRate = tf.placeholder(tf.float32, shape=[])
#beta = tf.placeholder(tf.float32, shape=[])
phase_train = tf.placeholder(tf.bool, name='phase_train')

W_conv1 = weight_variable([3, 3, 3, 24])
conv1_bn = convlayer_bn(x, W_conv1, 'VALID', phase_train)
h_conv1 = tf.nn.elu(conv1_bn)

W_conv2 = weight_variable([3, 3, 24, 24])
conv2_bn = convlayer_bn(h_conv1, W_conv2, 'VALID', phase_train)
h_conv2 = tf.nn.elu(conv2_bn)

W_conv3 = weight_variable([3, 3, 24, 24])
conv3_bn = convlayer_bn(h_conv2, W_conv3, 'VALID', phase_train)
h_conv3 = tf.nn.elu(conv3_bn)

W_fc = weight_variable([nz-3*2, nx-3*2, 24, 1])
b_fc1 = bias_variable([1])
h_fc1 = fc_layer(h_conv3, W_fc, b_fc1)

predict_y = h_fc1
predict_y = tf.reshape(predict_y, [-1, 1])
real_y = tf.reshape(y, [-1, 1])
# print(real_y)
# print(y)

cost_y = tf.reduce_mean(tf.square(predict_y - real_y))

cost_w = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
          tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_fc))

cost = cost_y + 0.0001*cost_w
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(LearnRate).minimize(cost)
#optimizer = tf.train.AdamOptimizer(LearnRate).minimize(cost)

saver = tf.train.Saver(max_to_keep=None)

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

###########################Copy Pasted from training.py##################

# learning iteration per same learning rate
subEpoch = 100
nStep = 5
totalEpoch = subEpoch * nStep
totalBatch = 100

# initial learning rate & lr decays to lr*1/5 per N_epoch
iniLR = 0.0005

stepCosts = []  
for i1 in range(nStep):
    # Change Learningrate
    rate = iniLR * np.power(5.0, -float(i1))
    subEpochCosts = []
    stepCosts.append(subEpochCosts)
    
    for i2 in range(subEpoch):
        batchCosts = []
        subEpochCosts.append(batchCosts)
        
        for i3 in range(totalBatch):
            x_data, y_data = ld.GetBatch(i3)
            #print(x_data.dtype , x_data.shape)
            #print(y_data.dtype , y_data.shape)
            # print(phase_train)
            print(i1, i2, i3)
            try:
                outPut, batchCost = sess.run([optimizer, cost], feed_dict={
                    x: x_data, y: y_data, LearnRate: rate, phase_train: True})
                print(outPut,batchCost)
                batchCosts.append(batchCost)
            except:
                info = sys.exc_info()
                print("Unexpected error:", info)
                raise
     
#plot cost
print("completed")
