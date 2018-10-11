import tensorflow as tf
import os
import logging
import numpy as np
import scipy as scp

from vgg_fcn import fcn8_vgg

# prefix = '../models/vgg_fcn8/vgg_fcn8'
# epoch = 18
# model_path = '%s-%s'%(prefix,epoch)
#
# print(model_path)
# model_dict = '/'.join(model_path.split('/')[:-1])
# print(model_dict)
# ckpt = tf.train.get_checkpoint_state(model_dict)


imgname = 'test_0.jpg'
img1 = scp.misc.imread(imgname)

model_path = '../models/vgg_fcn8/'



input_holder = tf.placeholder("float")

feed_dict = {input_holder: img1}

vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path='../vgg16.npy')
with tf.name_scope('vgg_fcn8'):
    vgg_fcn.build(input_holder, train=False)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver(max_to_keep=0)

saver.restore(sess, tf.train.latest_checkpoint(model_path))

up = sess.run(vgg_fcn.pred_img,feed_dict)

scp.misc.imsave('result.jpg', up)