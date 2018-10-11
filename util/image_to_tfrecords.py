import tensorflow as tf
import cv2
import os
import sys
data_dir = '../val_mesh_data'
target_dir = '../data'






def generate_tfrecords(data_dir,output_file,height,width):
    file_names = os.listdir(data_dir)
    tfrecord_writer = tf.python_io.TFRecordWriter(output_file)
    for i,name in enumerate(file_names):
        #read image

        sys.stdout.write('\r >>%d/%d images done' %(i+1,len(file_names)))
        sys.stdout.flush()
        img_path = os.path.join(data_dir, name)
        img_feature = _image_to_feature(img_path,height=height,width=width)

        #get target name and read the image
        target_name = name.split('mesh')[0] + '.jpg'
        target_path = os.path.join(target_dir, target_name)
        target_feature = _image_to_feature(target_path,height=height,width=width
                                           )

        example = _feature_to_example(img_feature,target_feature)
        tfrecord_writer.write(example.SerializeToString())


def _image_to_feature(img_path,height,width):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(width,height))
    img_data = img.tostring()
    # cv2.imshow('a', img)
    # cv2.waitKey(0)
    #print(img.shape)
    #print(len(img_data))
    return bytes_feature(img_data)

def _feature_to_example(img_feature,target_feature):
    return tf.train.Example(features =
                            tf.train.Features(feature=
                                              {'image/encoded':img_feature,
                                               'target/encoded':target_feature}))

def bytes_feature(value):
    '''
    Transfrom bytes into tensorflow feature
    :param value:
    :return: TF-Feature
    '''
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

if __name__ == '__main__':
    val_file = '../val_tfrecord/meshface_val.tfrecord'
    train_file = '../train_tfrecord/meshface_train.tfrecord'
    h = 220
    w = 178
    train_dir = '../train_mesh_data/'
    val_dir = '../val_mesh_data/'
    target_dir = '../data/'
    generate_tfrecords(data_dir=val_dir,output_file=val_file,height=h,width=w)
    #generate_tfrecords(data_dir=train_dir,output_file=train_file, height = h, width =w)




