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

from vgg_fcn import fcn8_vgg, vgg_fcn_small
import os

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

result_dir = '../test_output/vgg_fcn_small_epoch40'
img_dir = '../../DATA/val_mesh_data'
img_names = os.listdir(img_dir)


model_path = '../models/vgg_fcn_small/'
print(tf.train.latest_checkpoint(model_path))
with tf.Session() as sess:
    images = tf.placeholder("float")

    batch_images = tf.expand_dims(images, 0)

    vgg_fcn = vgg_fcn_small.VGG_FCN_SMALL(vgg16_npy_path='../vgg16.npy')
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images)



    print('Finished building Network.')

    logging.warning("Score weights are initialized random.")
    logging.warning("Do not expect meaningful results.")

    logging.info("Start Initializing Variabels.")

    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    print('Running the Network')

   # png_image = tf.image.encode_png
    tensors = [ vgg_fcn.pred_img]
    time0 = time.time()
    idx = 0
    for name in img_names:
        idx += 1
        img_name = os.path.join(img_dir,name)
        print(img_name)
        image = scp.misc.imread(img_name)
        print(image.shape)
        up = sess.run(tensors, feed_dict={images:image})
        up = np.squeeze(up)
        result_path = os.path.join(result_dir, name)
        print(result_path)
        scp.misc.imsave(result_path, up)

