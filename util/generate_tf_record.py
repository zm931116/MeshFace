from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import tensorflow as tf

from util import dataset_utils

_NUM_SHARDS = 10
TARGET_PATH = '../data/'
class ImageReader(object):

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _get_dataset_filename(dataset_dir, split_name, shard_id,num_shards):
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    print(type(shard_id))
    print('shard_id',shard_id)
    print(type(num_shards))
    output_filename = 'meshface_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, files_path,filenames, dataset_dir,num_shards):
    '''
    convert dataset into mesh and clear pair and store them in tfrecord files
    :param split_name:
    :param filenames: list of train image names
    :param files_path: path of the original images
    :param dataset_dir:
    :return:
    '''
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))
    with tf.Graph().as_default():
        image_reader = ImageReader()
        #image_reader1 = ImageReader()

        with tf.Session() as sess:
            for shard_id in range (num_shards):
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id,num_shards)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1)* num_per_shard,len(filenames))
                    for i in range (start_ndx,end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d'  % (
                            i + 1, len(filenames),shard_id))
                        sys.stdout.flush()

                        #read the train image:
                        image_path = os.path.join(files_path,filenames[i])
                        #print('\n',image_path)
                        image_data = tf.gfile.FastGFile(image_path,'rb').read()
                        height, width = image_reader.read_image_dims(sess,image_data)

                        #get target name and path and read target image
                        target_name = filenames[i].split('mesh')[0] + '.jpg'
                        target_path = os.path.join(TARGET_PATH,target_name)
                        #print('target path',target_path)
                        target_image_data = tf.gfile.FastGFile(target_path,'rb').read()
                        target_height , target_width = image_reader.read_image_dims(sess,target_image_data)

                        example = dataset_utils.image_pair_to_tfexample(image_data,target_image_data)
                        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__=='__main__':
    train_path = '../train_mesh_data/'
    val_path = '../val_mesh_data/'

    train_dataset_path = '../train_tfrecord/'
    val_dataset_path = '../val_tfrecord/'
    train_file_names = os.listdir(train_path)
    val_file_names = os.listdir(val_path)
    _convert_dataset('train',train_path,train_file_names,train_dataset_path,num_shards=10)
    _convert_dataset('validation',val_path,val_file_names,val_dataset_path,num_shards=1)

