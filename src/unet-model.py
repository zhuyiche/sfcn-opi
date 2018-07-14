from keras.models import Model
from keras.layers import Input, merge, core, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, concatenate,\
                         ZeroPadding2D, Activation, Reshape, Permute, Conv2DTranspose
from keras.activations import relu, softmax
from model import generator_without_augmentation
import util
from keras.optimizers import SGD, Adam
from util import LoadDataset
import numpy as np
from keras.utils import plot_model, np_utils

def generator_without_augmentation(features, labels, batch_size):
    batch_features = np.zeros((batch_size, 64, 64, 3))
    batch_labels = np.zeros((batch_size, 64, 64, 2))
    #print('features.shape: {} and {}'.format(features.shape[0], features.shape))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index = np.random.choice(features.shape[0], 1)
            #print('this is index: {}'.format(index))
            batch_features[i] = features[index]#some_processing(features[index])
            #print(batch_features[i].shape)
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels


def down_sample_block(input, filter, filter_size = (3,3),
                      pool_size = (2,2), dropout_rate = 0.2, activation = 'relu'):
    conv = Convolution2D(filter, 3, 3, activation=activation, border_mode='same')(input)
    conv = Dropout(dropout_rate)(conv)
    conv = Convolution2D(filter, 3, 3, activation=activation, border_mode='same')(conv)
    pool = MaxPooling2D(pool_size=pool_size)(conv)
    return pool, conv


def Unet(nClasses, input_width=64, input_height=64, nChannels=3):
    inputs = Input((input_height, input_width, nChannels))

    pool1, conv1 = down_sample_block(inputs, 32)
    pool2, conv2 = down_sample_block(inputs, 64)
    pool3, conv3 = down_sample_block(inputs, 128)

    up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)

    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)

    conv6 = Convolution2D(nClasses, 1, 1, activation='relu', border_mode='same')(conv5)
    conv6 = core.Reshape((nClasses, input_height * input_width))(conv6)
    conv6 = core.Permute((2, 1))(conv6)

    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)

    return model


def VGGUnet2(n_classes=4, input_height=64, input_width=64):

    assert input_height%32 == 0
    assert input_width%32 == 0

    # https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
    img_input = Input(shape=(input_height,input_width,3))

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(img_input)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[img_input], outputs=[outputs])
    return model

import os
#file_path = os.path.dirname( os.path.abspath(__file__) )
VGG_Weights_path = "/home/yichen/Desktop/" \
                   "sfcn-opi-yichen/vgg16_weights_th_dim_ordering_th_kernels.h5"


if __name__ == '__main__':
    p = '/home/yichen/Desktop/sfcn-opi-yichen/CRCHistoPhenotypes_2016_04_28/Cropping'
    path = '/Users/yichen/Desktop/CRCHistoPhenotypes_2016_04_28/Data_Augmentation'

    cls_path = '/home/yichen/Desktop/sfcn-opi-yichen/CRCHistoPhenotypes_2016_04_28/cls_and_det'

    weight_decay = 0.005
    epsilon = 1e-7
    epochs = 20
    BATCH_SIZE = 32
    d = LoadDataset(p)

    print('model summary starts')
    #a = SCFNnetwork()
    train_imgs, train_det_masks, train_cls_masks = d.load_data(p, type='train')
    print('didida')
    valid_imgs, valid_det_masks, valid_cls_masks = d.load_data(p, type='validation')
    test_imgs, test_det_masks, test_cls_masks = d.load_data(p, type='test')
    print(test_imgs.shape, test_det_masks.shape, test_cls_masks.shape)
    test_det = np_utils.to_categorical(test_det_masks, 2)
    test_cls = np_utils.to_categorical(test_cls_masks, 5)
    train_det = np_utils.to_categorical(train_det_masks, 2)
    train_cls = np_utils.to_categorical(train_cls_masks, 5)
    valid_det = np_utils.to_categorical(valid_det_masks, 2)
    valid_cls = np_utils.to_categorical(valid_cls_masks, 5)

    #PRF = PrecisionRecallF1Callback([valid_imgs, valid_det])
    print('train_imgs: {}, train_det: {}, train_cls: {}'.format(train_imgs.shape, train_det.shape, train_cls.shape))
    print('valid_imgs: {}, valid_det: {}, validn_cls: {}'.format(valid_imgs.shape, valid_det.shape, valid_cls.shape))
    print('test_imgs: {}, test_det: {}, test_cls: {}'.format(test_imgs.shape, test_det.shape, test_cls.shape))
    # det_model, ori_input, x_for_cls = a.detection_branch()

    det_model = VGGUnet2()
    det_model.summary()

    det_model.compile(optimizer=Adam(lr=0.0001),
                      loss = 'categorical_crossentropy', metrics=['accuracy'])

    #PRF.set_model(det_model)

    print('train_imgs.shape: {}, valid_imgs.shape: {}'.format(train_imgs.shape, valid_imgs.shape))
    print('train_det.shape-train_cls.shape: {}-{}'.format(train_det.shape, train_cls.shape))
    print('valid_det.shape-valid_cls.shape:{}-{}'.format(valid_det.shape, valid_cls.shape))
    print('detection branch training starts')
    # a ,b = generator_without_augmentation(train_imgs, train_det, batch_size=BATCH_SIZE)

    for epoch in range(epochs):
        det_model.fit_generator(generator_without_augmentation(train_imgs, train_det, batch_size=BATCH_SIZE), epochs=1,
                                steps_per_epoch=30,
                                validation_data=generator_without_augmentation(valid_imgs, valid_det, BATCH_SIZE),
                                validation_steps=5)

    score = det_model.evaluate(x=test_imgs, y=test_det_masks, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])