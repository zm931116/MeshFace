import tensorflow as tf
import cv2
import os
import sys
from PIL import Image
import time
import math

from train.fcn_config import config
from vgg_fcn import fcn8_vgg, loss, vgg_fcn_small, vgg_fcn_smallX


def read_tf_record(record_path, img_height, img_width, img_channel, batch_size, capacity_factor, reading_threads_num):
    filename_queue = tf.train.string_input_producer([record_path], shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    image_features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'target/encoded': tf.FixedLenFeature([], tf.string)
        }
    )
    image = tf.decode_raw(image_features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [img_height, img_width, img_channel])
    # image = image_color_distort(image)
    # image_float = tf.cast(image, tf.float32)

    target = tf.decode_raw(image_features['target/encoded'], tf.uint8)
    target = tf.reshape(target, [img_height, img_width, img_channel])
    # target = image_color_distort(target)
    # target = tf.cast(target, tf.float32)

    image_batch, target_batch = tf.train.shuffle_batch(
        [image, target],
        batch_size=batch_size,
        capacity=capacity_factor * batch_size,
        min_after_dequeue=batch_size,
        num_threads=reading_threads_num
    )

    return image_batch, target_batch


def _configure_optimizer(optimizer_name=None, learning_rate=config.lr):

    if optimizer_name == 'adagradDA':
        optimizer = tf.train.AdagradDAOptimizer(learning_rate, l2_regularization_strength=0.1)
    elif optimizer_name == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99)

    elif optimizer_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9)
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    return optimizer


def train(train_dir, val_dir, max_epoch, model_path, log_dir, start_epoch = 0,save_interval=2, val_interval=100, display=100,
          from_scrach=True):
    val_log_dir = log_dir + '_val'
    print('validation log path: ', val_log_dir)
    batch_size = config.batch_size
    img_height = config.img_height
    img_width = config.img_width
    img_channel = config.img_channel
    capacity_factor = config.capacity_factor
    reading_threads_num = config.reading_threads_num

    image_batch, target_batch = read_tf_record(train_dir, img_height, img_width, img_channel, batch_size,
                                               capacity_factor,
                                               reading_threads_num)

    val_image_batch, val_target_batch = read_tf_record(val_dir, img_height, img_width, img_channel, batch_size,
                                                       capacity_factor,
                                                       reading_threads_num)

    # create input node
    # image_shape = [batch_size, img_height, img_width, img_channel]
    input_holder = tf.placeholder(tf.float32, name='input_image')
    target_holder = tf.placeholder(tf.float32, name='target_image')
    global_step = tf.train.get_or_create_global_step()

    tf.summary.image('input', input_holder, max_outputs=2)
    tf.summary.image('input_target', target_holder, max_outputs=2)

    vgg_fcn = vgg_fcn_smallX.VGG_FCN_SMALL(vgg16_npy_path='../vgg16.npy')
    with tf.name_scope('vgg_fcn_smallX'):
        vgg_fcn.build(input_holder, train=True)

    l2_loss = loss.Euclidean_loss(vgg_fcn.pred_img, target_holder) / batch_size
    optimizer = _configure_optimizer(optimizer_name='adam')
    train_op = optimizer.minimize(l2_loss, global_step)
    tf.summary.image('prediction', vgg_fcn.pred_img, max_outputs=2)
    tf.summary.scalar('Euclidean_loss', l2_loss)
    summary_op = tf.summary.merge_all()
    train_fetches = [vgg_fcn.pred_img, train_op]
    val_fetches = [vgg_fcn.pred_img, l2_loss, summary_op]

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    # writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer_val = tf.summary.FileWriter(val_log_dir, sess.graph)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step_per_epoch = int(math.ceil(27000 / batch_size))
    MAX_STEP = max_epoch * step_per_epoch
    epoch = 0
    print('max step is', MAX_STEP)
    time0 = time.time()

    saver = tf.train.Saver(max_to_keep=0)
    print(model_path)
    print(sess.run(global_step.read_value()))
    if not from_scrach:
        epoch = start_epoch
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

    start_step = epoch * step_per_epoch
    i = 0
    try:
        for step in range(start_step, MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break

            # print('im here')
            # inference
            image_batch_array, target_batch_array = sess.run([image_batch, target_batch])

            prediction, _ = sess.run(fetches=train_fetches,
                                     feed_dict={input_holder: image_batch_array, target_holder: target_batch_array})
            if (step ) % display == 0:
                print('step--', step + 1)
                print('time cost for last 100 steps:', time.time() - time0)
                # print(prediction.shape)
                # writer.add_summary(summary, global_step=step)
                time0 = time.time()

                # validation process
                time_val = time.time()
                image_batch_val, target_batch_val = sess.run([val_image_batch, val_target_batch])

                val_prediction, val_l2_loss, val_summary = sess.run(fetches=val_fetches,
                                                                    feed_dict={input_holder: image_batch_val,
                                                                               target_holder: target_batch_val})
                writer_val.add_summary(val_summary, global_step=step+1)
                # print('validation!!')
                # print(val_prediction.shape)
                time_val1 = time.time() - time_val

                print('validation loss:', val_l2_loss)
                print('validation time: ', time_val1)
            # if (step+1) % val_interval == 0:
            #     pass
            # if step == 0:
            #     print('saving the model')
            #     model_path = os.path.join(model_path,'initial')
            #     saved_path = saver.save(sess, model_path, global_step=epoch)
            #     print('model had been saved at:', saved_path)
            if i * batch_size > 27000 * 5:
                epoch = epoch + 5
                i = 0
                model_name = os.path.join(model_path, 'vgg_smallX')
                saved_path = saver.save(sess, model_name, global_step=epoch)
                print('model had been saved at:', saved_path)

            # print(image_batch_array[0])

            # img = Image.fromarray(image_batch_array[0],'RGB')
            # img.save('../test_output/%d.jpg'% step)
            # target_img = Image.fromarray(target_batch_array[0],'RGB')
            # target_img.save('../test_output/%dt.jpg'% step)



    except tf.errors.OutOfRangeError:
        print('finished')

    finally:
        coord.request_stop()
        # writer.close()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    train_record_path = '../train_tfrecord/meshface_train.tfrecord'
    val_tfrecord_path = '../val_tfrecord/meshface_val.tfrecord'
    model_path = '../models/vgg_fcn_smallX/'
    log_dir = '../logs/vgg_fcn_smallX'
    val_log_dir = ''
    train(train_dir=train_record_path, val_dir=val_tfrecord_path,start_epoch=80, max_epoch=100,
          model_path=model_path, log_dir=log_dir,
          display=100, from_scrach=False)
