import tensorflow as tf
import os
import logging
import numpy as np
import scipy as scp

from vgg_fcn import fcn8_vgg

a = tf.constant(1,dtype=tf.int32,shape = [2,3,3,4],name = 'a')
b = tf.constant(1,dtype=tf.int32,shape = [2,3,3,4],name = 'b')
c = tf.concat([a,b],axis = -1)

sess = tf.Session()
output =sess.run(c)
print(output.shape)