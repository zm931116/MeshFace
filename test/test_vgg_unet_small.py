#!/usr/bin/env python
import cv2
import logging
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf

from vgg_fcn import vgg_unet_small
import os

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

result_dir = '../test_output/vgg_unet18_epoch60'
img_dir = '../../DATA/val_mesh_data'
img_names = os.listdir(img_dir)

#
model_path = '../models/vgg_unet18/'
print('##################checkpoint is :',tf.train.latest_checkpoint(model_path))
with tf.Session() as sess:
    images = tf.placeholder("float")

    batch_images = tf.expand_dims(images, 0)

    vgg_unet = vgg_unet_small.VGG_UNET_SMALL(vgg16_npy_path='../vgg16.npy')
    with tf.name_scope("content_vgg"):
        vgg_unet.build(batch_images)



    print('Finished building Network.')

    logging.info("Start Initializing Variabels.")

    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    print('Running the Network')

   # png_image = tf.image.encode_png
    tensors = [ vgg_unet.pred_img]
    layers = [vgg_unet.conv3_2,vgg_unet.pool3,vgg_unet.conv4_1,vgg_unet.conv4_2]
    time0 = time.time()
    idx = 0
    for name in img_names:
        idx += 1
        img_name = os.path.join(img_dir,name)
        print(img_name)
        image = scp.misc.imread(img_name)
        print(image.shape)
        # layer_output = sess.run(layers,feed_dict={images:image})
        # for layer in layer_output:
        #     print(layer.shape)
        up = sess.run(tensors, feed_dict={images:image})
        up = np.squeeze(up)
        result_path = os.path.join(result_dir, name)
        print(result_path)
        scp.misc.imsave(result_path, up)
    forward_time = (time.time() - time0)
    s_time = forward_time/50
    print('total time %2.4f, single image time: %2.4f'%(forward_time,s_time))
