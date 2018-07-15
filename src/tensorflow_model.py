import os
import cv2
import tensorflow as tf
from util import load_data
from keras.utils import plot_model,np_utils
#from util import next_batch
import numpy as np
'''
class SCFNnetwork():
    def __init__(self, input_shape = (64, 64, 3)):
        self.input_shape = input_shape

    def batch_norm(self, inputs, training):
        return tf.layers.batch_normalization(inputs=inputs, axis=3,
                                            training=training, fused=True)

    def first_layer(self, inputs, filters, kernel_size, batch_norm_training_time = True):
        x = tf.layers.conv2d(inputs=inputs,filters=filters,
                             kernel_size=(kernel_size, kernel_size), kernel_initializer=tf.glorot_uniform_initializer(),
                             strides=(1, 1), padding='SAME')

        x = self.batch_norm(x, training=batch_norm_training_time)
        outputs = tf.nn.relu(x)
        return outputs

    def convolution_block(self, inputs, filters, kernel_size,
                          stage, block, batch_norm_training_time = True):
        x = tf.layers.conv2d(inputs=inputs, filters = filters, padding='SAME', strides=(2,2),
                             kernel_size=(kernel_size,kernel_size), kernel_initializer=tf.variance_scaling_initializer())
        x = self.batch_norm(x, training=batch_norm_training_time)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(inputs=x, filters=filters, padding='SAME', strides=(1,1),
                             kernel_size=(kernel_size,kernel_size), kernel_initializer=tf.variance_scaling_initializer())
        x = self.batch_norm(x, training=batch_norm_training_time)

        x_shortcut = tf.layers.conv2d(inputs=inputs, filters = filters, kernel_size=(1,1),
                                      strides=(2,2), padding='SAME',
                                      kernel_initializer=tf.variance_scaling_initializer())
        x_shortcut = self.batch_norm(inputs=x_shortcut, training=batch_norm_training_time)

        x += x_shortcut
        outputs = tf.nn.relu(x)
        return outputs

    def identity_block(self, inputs, filters, kernel_size,
                          stage, block, batch_norm_training_time=True):
        x = tf.layers.conv2d(inputs=inputs, filters=filters, padding='SAME', strides=(1,1),
                             kernel_size=(kernel_size,kernel_size), kernel_initializer=tf.variance_scaling_initializer())
        x = self.batch_norm(x, training=batch_norm_training_time)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(inputs=x, filters=filters, padding='SAME', strides=(1,1),
                             kernel_size=(kernel_size,kernel_size), kernel_initializer=tf.variance_scaling_initializer())
        x = self.batch_norm(x, training=batch_norm_training_time)

        x += inputs
        outputs = tf.nn.relu(x)
        return outputs

    def res_block(self, inputs, filters, kernel_size, stages, block, if_conv = False):
        x = inputs
        if not if_conv:
            for stage in range(stages):
                x = self.identity_block(filters=filters, kernel_size = kernel_size,
                                        stage=stage, block=block, inputs=x)
        else:
            for stage in range(stages):
                if stage == 0:
                    x = self.convolution_block(filters =filters, kernel_size=kernel_size,
                                               stage=stage, block=block, inputs=inputs)
                else:
                    x = self.identity_block(filters =filters, kernel_size=kernel_size,
                                            stage=stage, block=block, inputs=x)
        return x

    def first_and_second_res_blocks(self, inputs, first_filter, second_filter, kernel_size):
        """
        Shared residual blocks for detection and classification layers.
        """
        with tf.name_scope('res_block_one'):
            x = self.res_block(inputs, filters=first_filter, kernel_size=kernel_size, stages=9, block=1)
        with tf.name_scope('res_block_two'):
            x = self.res_block(x, filters=second_filter, kernel_size=kernel_size, stages=9, block=2, if_conv=True)
        return x

    def third_res_blocks(self, inputs, filters, kernel_size, block):
        x = inputs
        x = self.res_block(x, filters=filters, kernel_size=kernel_size, stages=9, block=block, if_conv=True)
        return x

    def detection_branch(self, inputs, kernel_size = 3, batch_norm_training_time = True):
        #input_img = Input(shape=self.input_shape)
        with tf.name_scope('detection_branch'):
            with tf.name_scope('first_layer'):
                x = self.first_layer(inputs=inputs, kernel_size = kernel_size, filters=32)
            x = self.first_and_second_res_blocks(x, 32, 64, kernel_size)

            with tf.name_scope('divergent_path_one'):
                x_divergent_one = tf.layers.conv2d(inputs=x, filters=2, kernel_size=(1, 1), padding='SAME',
                                                   kernel_initializer=tf.variance_scaling_initializer())
                x_divergent_one = self.batch_norm(inputs=x_divergent_one, training=batch_norm_training_time)
                x_divergent_one = tf.nn.relu(x_divergent_one)
            with tf.name_scope('divergent_path_two'):

                x_for_future_classification = self.third_res_blocks(inputs=x, filters=128, kernel_size=3, block=4)

                #these are detection branch
                x_divergent_two = tf.layers.conv2d(inputs=x_for_future_classification,
                                                   filters=2, kernel_size=(1, 1), padding='SAME',
                                                   kernel_initializer=tf.variance_scaling_initializer())
                x_divergent_two = self.batch_norm(inputs=x_divergent_two, training=batch_norm_training_time)
                x_divergent_two = tf.nn.relu(x_divergent_two)
                x_divergent_two = tf.layers.conv2d_transpose(inputs=x_divergent_two,
                                                             filters=2, kernel_size=(3, 3),
                                                             strides=(2, 2), padding='SAME',
                                                             kernel_initializer=tf.variance_scaling_initializer())
                x_divergent_two = self.batch_norm(x_divergent_two, training=batch_norm_training_time)
                x_divergent_two = tf.nn.relu(x_divergent_two)
            with tf.name_scope('merge_divergent_path'):
                print('divergenet_one: {}, divergent_two: {}'.format(x_divergent_one.shape, x_divergent_two.shape))
                x_merge = x_divergent_one + x_divergent_two
                x_detection = tf.layers.conv2d_transpose(inputs=x_merge, filters=2,
                                                         kernel_size=(3, 3), strides=(2, 2), padding='SAME',
                                                         kernel_initializer=tf.variance_scaling_initializer())
                x_detection = self.batch_norm(x_detection, training=batch_norm_training_time)
                x_detection = tf.nn.softmax(x_detection)
        return x_detection#, x_for_future_classification


if __name__ == '__main__':
    p = '/home/yichen/Desktop/sfcn-opi-yichen/Cropping'
    path = '/Users/yichen/Desktop/CRCHistoPhenotypes_2016_04_28/Data_Augmentation'
    weight_decay = 0.005
    epsilon = 1e-7
    training_epochs = 10
    BATCH_SIZE = 32

    a = SCFNnetwork(p)
    train_imgs, train_det_masks, train_cls_masks = load_data(p, type='train')
    valid_imgs, valid_det_masks, valid_cls_masks = load_data(p, type='validation')
    train_det = np_utils.to_categorical(train_det_masks, 2)
    train_cls = np_utils.to_categorical(train_cls_masks, 5)
    valid_det = np_utils.to_categorical(valid_det_masks, 2)
    valid_cls = np_utils.to_categorical(valid_cls_masks, 5)

    train_imgs_holder = tf.placeholder(tf.float32, [None, 64, 64, 3])
    train_det_masks_holder = tf.placeholder(tf.float32, [None, 64, 64, 2])
    y_pred = a.detection_branch(train_imgs_holder)
    y = tf.placeholder(tf.float16, [None, 64, 64, 2])
    print(train_imgs.shape)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print('training start')
    cross_entropy = -tf.reduce_sum(train_det_masks_holder * tf.log(y_pred))
    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
    print('after optimizer')
    corrected_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(corrected_prediction, 'float'))
    print('before acc')
    for epoch in range(training_epochs):
        total_train_batch = int(len(train_imgs) / BATCH_SIZE)
        print('epoch: {}'.format(epoch))
        for i in range(total_train_batch):
            print('i: {}'.format(i))
            rand = np.random.randint(0, train_imgs.shape[0])
            batch_xs, batch_ys = train_imgs[rand: rand+BATCH_SIZE, :, :, :], train_det[rand: rand+BATCH_SIZE, :, :, :]
            print(type(batch_xs), batch_ys.shape)
            #print(batch_xs)
            train_acc = sess.run([optimizer, accuracy], feed_dict={train_imgs_holder: batch_xs, train_det_masks_holder: batch_ys})[1]
        print('valid::: ')
        total_validation_batch = int(len(valid_imgs) / BATCH_SIZE)
        for i in range(total_validation_batch):
            batch_xs, batch_ys = next_batch(batch_size=BATCH_SIZE,
                                            data=valid_imgs,
                                            labels=valid_det)
            acc = sess.run([accuracy], feed_dict={
                train_imgs_holder: batch_xs,
                train_det_masks_holder: batch_ys,
            })
            '''
        #print('epoch: {}, train: {}, valid: {}'.format(epoch, train_acc, acc)) ''