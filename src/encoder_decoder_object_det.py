import numpy as np
import tensorflow as tf
import keras.backend as K
import keras
from keras.layers import Input,Conv2D,Add,BatchNormalization,Activation, Lambda, Multiply, Conv2DTranspose, Concatenate, ZeroPadding2D
from keras.layers.convolutional import AtrousConv2D
from keras.models import Model
from keras.utils import plot_model,np_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, LearningRateScheduler, \
    TensorBoard,ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.regularizers import l2
from util import load_data
import os, time
from image_augmentation import ImageCropping
from imgaug import augmenters as iaa
import imgaug as ia
from loss import detection_focal_loss_K, \
    detection_loss_without_part2_K, detection_loss_K, \
    detection_focal_loss_without_part2_K, detection_double_focal_loss_K,\
    detection_double_focal_loss_indicator_K
from config import Config
from tensorflow.python.client import device_lib
weight_decay = 0.005
epsilon = 1e-7

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'cls_and_det')
TENSORBOARD_DIR = os.path.join(ROOT_DIR, 'tensorboard_logs')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoint')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'model_weights')

class Conv3l2(keras.layers.Conv2D):
    """
    Custom convolution layer with default 3*3 kernel size and L2 regularization.
    Default padding change to 'same' in this case.
    """
    def __init__(self, filters, kernel_regularizer_weight,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(self.__class__, self).__init__(filters,
                                             kernel_size=(3, 3),
                                             strides=strides,
                                             padding=padding,
                                             data_format=data_format,
                                             dilation_rate=dilation_rate,
                                             activation=activation,
                                             use_bias=use_bias,
                                             kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer,
                                             kernel_regularizer=l2(kernel_regularizer_weight),
                                             bias_regularizer=bias_regularizer,
                                             activity_regularizer=activity_regularizer,
                                             kernel_constraint=kernel_constraint,
                                             bias_constraint=bias_constraint,
                                             **kwargs)


class Detnet:
    """
    Backbone of SFCN-OPI.
    """
    def __init__(self, input_shape=(512, 512, 3)):
        # self.inputs = inputs
        self.input_shape = input_shape
        self.l2r = 0.001

    def first_layer(self, inputs, trainable=True):
        """
        First convolution layer.
        """
        x = Conv3l2(filters=32, name='Conv_1',
                    kernel_regularizer_weight=self.l2r,
                    trainable=trainable)(inputs)
        x = BatchNormalization(name='BN_1',trainable=trainable)(x)
        x = Activation('relu', name='act_1',trainable=trainable)(x)
        return x


    ###########################################
    # ResNet Graph
    ###########################################
    def identity_block_3(self, f, stage, block, inputs, trainable=True):
        """
        :param f: number of filters
        :param stage: stage of residual blocks
        :param block: ith module
        :param trainable: freeze layer if false
        """
        x_shortcut = inputs

        x = Conv3l2(filters=f, kernel_regularizer_weight=self.l2r,
                    name=str(stage) + '_' + str(block) + '_idblock_conv_1',
                    trainable=trainable)(inputs)
        x = BatchNormalization(name=str(stage) + '_' + str(block) +'_idblock_BN_1',
                               trainable=trainable)(x)
        x = Activation('relu', name=str(stage) + '_' + str(block)+ '_idblock_act_1',
                       trainable=trainable)(x)

        x = Conv3l2(filters=f, kernel_regularizer_weight=self.l2r,
                    name=str(stage) + '_' + str(block)+ '_idblock_conv_2')(x)
        x = BatchNormalization(name=str(stage) + '_' + str(block) + '_idblock_BN_2',
                               trainable=trainable)(x)
        x = Activation('relu', name=str(stage) + '_' + str(block) + '_idblock_act_2',
                       trainable=trainable)(x)

        x = Conv3l2(filters=f, kernel_regularizer_weight=self.l2r,
                    name=str(stage) + '_' + str(block) + '_idblock_conv_3',
                    trainable=trainable)(x)
        x_id_3 = BatchNormalization(name=str(stage) + '_' + str(block)+ '_idblock_BN_3',
                               trainable=trainable)(x)


        x_add = Add(name=str(stage) + '_' + str(block)+ '_idblock_add',trainable=trainable)([x_id_3, x_shortcut])
        x_idblock_output = Activation('relu', name=str(stage) + '_' + str(block)+ '_idblock_act_outout',
                       trainable=trainable)(x_add)
        return x_idblock_output

    def convolution_block_3(self, f, stage, block, inputs, trainable=True):
        """
        :param f: number of filters
        :param stage: stage of residual blocks
        :param block: ith module
        """
        x = Conv3l2(filters=f, strides=(2,2), kernel_regularizer_weight=self.l2r,
                    name=str(stage) + str(block)+'_' + '_convblock_conv_1',
                    trainable=trainable)(inputs)
        x = BatchNormalization(name= str(stage) + '_' + str(block)+ '_convblock_BN_1',
                               trainable=trainable)(x)
        x = Activation('relu', name=str(stage) + '_' + str(block) + '_convblock_act_1',
                       trainable=trainable)(x)
        x = Conv3l2(filters=f, kernel_regularizer_weight=self.l2r,
                   name=str(stage) + '_' + str(block) + '_convblock_conv_2',
                    trainable=trainable)(x)
        x = BatchNormalization(name=str(stage) + '_' + str(block)+ '_convblock_BN_2',
                               trainable=trainable)(x)
        x = Activation('relu', name=str(stage) + '_' + str(block) + '_convblock_act_2',
                       trainable=trainable)(x)

        x = Conv3l2(filters=f, kernel_regularizer_weight=self.l2r,
                    name=str(stage) + '_' + str(block) + '_convblock_conv_3',
                    trainable=trainable)(x)
        x_conv_3 = BatchNormalization(name=str(stage) + '_' + str(block) + '_convblock_BN_3',
                               trainable=trainable)(x)

        x_shortcut = Conv2D(f, kernel_size=(1,1), strides=(2,2), padding='same',
                            kernel_regularizer=keras.regularizers.l2(self.l2r),
                            name=str(stage) + '_' + str(block) + '_convblock_shortcut_conv',
                            trainable=trainable)(inputs)
        x_shortcut = BatchNormalization(name =str(stage) + '_' +  str(block) + '_convblock_shortcut_BN_1',
                                        trainable=trainable)(x_shortcut)
        x_add = Add(name=str(stage) + '_' + str(block) + '_convblock_add',
                trainable=trainable)([x_conv_3, x_shortcut])
        x_convblock_output = Activation('relu', name=str(stage) + '_' + str(block) + '_convblock_act_output',
                       trainable=trainable)(x_add)
        return x_convblock_output
    #############################
    # Resnet 50
    #############################
    def resnet_50(self, inputs, filters, stages, trainable=True):
        x = inputs
        x_conv_2 = self.convolution_block_3(f=filters[0], stage=stages[0], block=1, inputs=x, trainable=trainable)
        x_id_21 = self.identity_block_3(f=filters[0], stage=stages[0], block=2, inputs=x_conv_2, trainable=trainable)
        x_id_22 = self.identity_block_3(f=filters[0], stage=stages[0], block=3, inputs=x_id_21, trainable=trainable)

        x_conv_3 = self.convolution_block_3(f=filters[1], stage=stages[1], block=1, inputs=x_id_22, trainable=trainable)
        x_id_31 = self.identity_block_3(f=filters[1], stage=stages[1], block=2, inputs=x_conv_3, trainable=trainable)
        x_id_32 = self.identity_block_3(f=filters[1], stage=stages[1], block=3, inputs=x_id_31, trainable=trainable)
        x_id_33 = self.identity_block_3(f=filters[1], stage=stages[1], block=4, inputs=x_id_32, trainable=trainable)

        x_conv_4 = self.convolution_block_3(f=filters[2], stage=stages[2], block=1, inputs=x_id_33, trainable=trainable)
        x_id_41 = self.identity_block_3(f=filters[2], stage=stages[2], block=2, inputs=x_conv_4, trainable=trainable)
        x_id_42 = self.identity_block_3(f=filters[2], stage=stages[2], block=3, inputs=x_id_41, trainable=trainable)
        x_id_43 = self.identity_block_3(f=filters[2], stage=stages[2], block=4, inputs=x_id_42, trainable=trainable)
        x_id_44 = self.identity_block_3(f=filters[2], stage=stages[2], block=5, inputs=x_id_43, trainable=trainable)
        x_id_45 = self.identity_block_3(f=filters[2], stage=stages[2], block=6, inputs=x_id_44, trainable=trainable)
        return x_id_22, x_id_33, x_id_45

    #############################
    # Resnet 101
    #############################
    def resnet_101(self, inputs, filters, stages, trainable=True):
        x = inputs
        x_conv_2 = self.convolution_block_3(f=filters[0], stage=stages[0], block=1, inputs=x, trainable=trainable)
        x_id_21 = self.identity_block_3(f=filters[0], stage=stages[0], block=2, inputs=x_conv_2, trainable=trainable)
        x_id_22 = self.identity_block_3(f=filters[0], stage=stages[0], block=3, inputs=x_id_21, trainable=trainable)

        x_conv_3 = self.convolution_block_3(f=filters[1], stage=stages[1], block=1, inputs=x_id_22, trainable=trainable)
        x_id_31 = self.identity_block_3(f=filters[1], stage=stages[1], block=2, inputs=x_conv_3, trainable=trainable)
        x_id_32 = self.identity_block_3(f=filters[1], stage=stages[1], block=3, inputs=x_id_31, trainable=trainable)
        x_id_33 = self.identity_block_3(f=filters[1], stage=stages[1], block=4, inputs=x_id_32, trainable=trainable)

        x_conv_4 = self.convolution_block_3(f=filters[2], stage=stages[2], block=1, inputs=x_id_33, trainable=trainable)
        x_id_41 = self.identity_block_3(f=filters[2], stage=stages[2], block=2, inputs=x_conv_4, trainable=trainable)
        x_id_42 = self.identity_block_3(f=filters[2], stage=stages[2], block=3, inputs=x_id_41, trainable=trainable)
        x_id_43 = self.identity_block_3(f=filters[2], stage=stages[2], block=4, inputs=x_id_42, trainable=trainable)
        x_id_44 = self.identity_block_3(f=filters[2], stage=stages[2], block=5, inputs=x_id_43, trainable=trainable)
        x_id_45 = self.identity_block_3(f=filters[2], stage=stages[2], block=6, inputs=x_id_44, trainable=trainable)
        x_id_46 = self.identity_block_3(f=filters[2], stage=stages[2], block=7, inputs=x_id_45, trainable=trainable)
        x_id_47 = self.identity_block_3(f=filters[2], stage=stages[2], block=8, inputs=x_id_46, trainable=trainable)
        x_id_48 = self.identity_block_3(f=filters[2], stage=stages[2], block=9, inputs=x_id_47, trainable=trainable)
        x_id_49 = self.identity_block_3(f=filters[2], stage=stages[2], block=10, inputs=x_id_48, trainable=trainable)
        x_id_410 = self.identity_block_3(f=filters[2], stage=stages[2], block=11, inputs=x_id_49, trainable=trainable)
        x_id_411 = self.identity_block_3(f=filters[2], stage=stages[2], block=12, inputs=x_id_410, trainable=trainable)
        x_id_412 = self.identity_block_3(f=filters[2], stage=stages[2], block=13, inputs=x_id_411, trainable=trainable)
        x_id_413 = self.identity_block_3(f=filters[2], stage=stages[2], block=14, inputs=x_id_412, trainable=trainable)
        x_id_414 = self.identity_block_3(f=filters[2], stage=stages[2], block=15, inputs=x_id_413, trainable=trainable)
        x_id_415 = self.identity_block_3(f=filters[2], stage=stages[2], block=16, inputs=x_id_414, trainable=trainable)
        x_id_416 = self.identity_block_3(f=filters[2], stage=stages[2], block=17, inputs=x_id_415, trainable=trainable)
        x_id_417 = self.identity_block_3(f=filters[2], stage=stages[2], block=18, inputs=x_id_416, trainable=trainable)
        x_id_418 = self.identity_block_3(f=filters[2], stage=stages[2], block=19, inputs=x_id_417, trainable=trainable)
        x_id_419 = self.identity_block_3(f=filters[2], stage=stages[2], block=20, inputs=x_id_418, trainable=trainable)
        x_id_420 = self.identity_block_3(f=filters[2], stage=stages[2], block=21, inputs=x_id_419, trainable=trainable)
        x_id_421 = self.identity_block_3(f=filters[2], stage=stages[2], block=22, inputs=x_id_420, trainable=trainable)
        x_id_422 = self.identity_block_3(f=filters[2], stage=stages[2], block=23, inputs=x_id_421, trainable=trainable)

        return x_id_22, x_id_33, x_id_422

    #############################
    # Resnet 152
    #############################
    def resnet_152(self, inputs, filters, stages, trainable=True):
        x = inputs
        x_conv_2 = self.convolution_block_3(f=filters[0], stage=stages[0], block=1, inputs=x, trainable=trainable)
        x_id_21 = self.identity_block_3(f=filters[0], stage=stages[0], block=2, inputs=x_conv_2, trainable=trainable)
        x_id_22 = self.identity_block_3(f=filters[0], stage=stages[0], block=3, inputs=x_id_21, trainable=trainable)

        x_conv_3 = self.convolution_block_3(f=filters[1], stage=stages[1], block=1, inputs=x_id_22, trainable=trainable)
        x_id_31 = self.identity_block_3(f=filters[1], stage=stages[1], block=2, inputs=x_conv_3, trainable=trainable)
        x_id_32 = self.identity_block_3(f=filters[1], stage=stages[1], block=3, inputs=x_id_31, trainable=trainable)
        x_id_33 = self.identity_block_3(f=filters[1], stage=stages[1], block=4, inputs=x_id_32, trainable=trainable)
        x_id_34 = self.identity_block_3(f=filters[1], stage=stages[1], block=5, inputs=x_id_33, trainable=trainable)
        x_id_35 = self.identity_block_3(f=filters[1], stage=stages[1], block=6, inputs=x_id_34, trainable=trainable)
        x_id_36 = self.identity_block_3(f=filters[1], stage=stages[1], block=7, inputs=x_id_35, trainable=trainable)
        x_id_37 = self.identity_block_3(f=filters[1], stage=stages[1], block=8, inputs=x_id_36, trainable=trainable)

        x_conv_4 = self.convolution_block_3(f=filters[2], stage=stages[2], block=1, inputs=x_id_37, trainable=trainable)
        x_id_41 = self.identity_block_3(f=filters[2], stage=stages[2], block=2, inputs=x_conv_4, trainable=trainable)
        x_id_42 = self.identity_block_3(f=filters[2], stage=stages[2], block=3, inputs=x_id_41, trainable=trainable)
        x_id_43 = self.identity_block_3(f=filters[2], stage=stages[2], block=4, inputs=x_id_42, trainable=trainable)
        x_id_44 = self.identity_block_3(f=filters[2], stage=stages[2], block=5, inputs=x_id_43, trainable=trainable)
        x_id_45 = self.identity_block_3(f=filters[2], stage=stages[2], block=6, inputs=x_id_44, trainable=trainable)
        x_id_46 = self.identity_block_3(f=filters[2], stage=stages[2], block=7, inputs=x_id_45, trainable=trainable)
        x_id_47 = self.identity_block_3(f=filters[2], stage=stages[2], block=8, inputs=x_id_46, trainable=trainable)
        x_id_48 = self.identity_block_3(f=filters[2], stage=stages[2], block=9, inputs=x_id_47, trainable=trainable)
        x_id_49 = self.identity_block_3(f=filters[2], stage=stages[2], block=10, inputs=x_id_48, trainable=trainable)
        x_id_410 = self.identity_block_3(f=filters[2], stage=stages[2], block=11, inputs=x_id_49, trainable=trainable)
        x_id_411 = self.identity_block_3(f=filters[2], stage=stages[2], block=12, inputs=x_id_410, trainable=trainable)
        x_id_412 = self.identity_block_3(f=filters[2], stage=stages[2], block=13, inputs=x_id_411, trainable=trainable)
        x_id_413 = self.identity_block_3(f=filters[2], stage=stages[2], block=14, inputs=x_id_412, trainable=trainable)
        x_id_414 = self.identity_block_3(f=filters[2], stage=stages[2], block=15, inputs=x_id_413, trainable=trainable)
        x_id_415 = self.identity_block_3(f=filters[2], stage=stages[2], block=16, inputs=x_id_414, trainable=trainable)
        x_id_416 = self.identity_block_3(f=filters[2], stage=stages[2], block=17, inputs=x_id_415, trainable=trainable)
        x_id_417 = self.identity_block_3(f=filters[2], stage=stages[2], block=18, inputs=x_id_416, trainable=trainable)
        x_id_418 = self.identity_block_3(f=filters[2], stage=stages[2], block=19, inputs=x_id_417, trainable=trainable)
        x_id_419 = self.identity_block_3(f=filters[2], stage=stages[2], block=20, inputs=x_id_418, trainable=trainable)
        x_id_420 = self.identity_block_3(f=filters[2], stage=stages[2], block=21, inputs=x_id_419, trainable=trainable)
        x_id_421 = self.identity_block_3(f=filters[2], stage=stages[2], block=22, inputs=x_id_420, trainable=trainable)
        x_id_422 = self.identity_block_3(f=filters[2], stage=stages[2], block=23, inputs=x_id_421, trainable=trainable)
        x_id_423 = self.identity_block_3(f=filters[2], stage=stages[2], block=24, inputs=x_id_422, trainable=trainable)
        x_id_424 = self.identity_block_3(f=filters[2], stage=stages[2], block=25, inputs=x_id_423, trainable=trainable)
        x_id_425 = self.identity_block_3(f=filters[2], stage=stages[2], block=26, inputs=x_id_424, trainable=trainable)
        x_id_426 = self.identity_block_3(f=filters[2], stage=stages[2], block=27, inputs=x_id_425, trainable=trainable)
        x_id_427 = self.identity_block_3(f=filters[2], stage=stages[2], block=28, inputs=x_id_426, trainable=trainable)
        x_id_428 = self.identity_block_3(f=filters[2], stage=stages[2], block=29, inputs=x_id_427, trainable=trainable)
        x_id_429 = self.identity_block_3(f=filters[2], stage=stages[2], block=30, inputs=x_id_428, trainable=trainable)
        x_id_430 = self.identity_block_3(f=filters[2], stage=stages[2], block=31, inputs=x_id_429, trainable=trainable)
        x_id_431 = self.identity_block_3(f=filters[2], stage=stages[2], block=32, inputs=x_id_430, trainable=trainable)
        x_id_432 = self.identity_block_3(f=filters[2], stage=stages[2], block=33, inputs=x_id_431, trainable=trainable)
        x_id_433 = self.identity_block_3(f=filters[2], stage=stages[2], block=34, inputs=x_id_432, trainable=trainable)
        x_id_434 = self.identity_block_3(f=filters[2], stage=stages[2], block=35, inputs=x_id_433, trainable=trainable)
        x_id_435 = self.identity_block_3(f=filters[2], stage=stages[2], block=36, inputs=x_id_434, trainable=trainable)

        return x_id_22, x_id_37, x_id_435

    ###################
    # Dilated Block
    ###################
    def dilated_bottleneck(self, inputs, stage, block):
        """
        Dilated block without 1x1 convolution projection, structure like res-id-block
        """
        x_shortcut = inputs
        x = Conv2D(filters=256, kernel_size=(1,1), padding='same', kernel_regularizer=l2(self.l2r),
                   name=str(stage) + '_' + str(block) + '_1' + '_dilated_block_first1x1')(inputs)
        x = BatchNormalization(name=str(stage) + '_' + str(block) + '_1'+ '_dilated_block_firstBN')(x)
        x = Activation('relu', name=str(stage) + '_' + str(block) + '_1'+ '_dilated_block_firstRELU')(x)

        x_dilated = Conv2D(filters=256, kernel_size=(3,3), padding='same',
                           name=str(stage) + '_' + str(block) + '_2' + '_dilated_block_dilatedconv',
                           kernel_regularizer=l2(self.l2r), dilation_rate=(2,2))(x)
        x_dilated = BatchNormalization(name=str(stage) + '_'+ str(block) + '_2'+ '_dilated_block_dilatedBN')(x_dilated)
        x_dilated = Activation('relu',name=str(stage) +'_'+ str(block) + '_2'+ '_dilated_block_dilatedRELU')(x_dilated)

        x_more = Conv2D(filters=256, kernel_size=(1,1), padding='same', kernel_regularizer=l2(self.l2r),
                        name=str(stage) + '_'+ str(block) + '_3'+ '_dilated_block_second1x1')(x_dilated)
        x_more = BatchNormalization(name=str(stage) + '_' + str(block) + '_3'+ '_dilated_block_secondBN')(x_more)
        x_more = Activation('relu', name=str(stage) + str(block) + '_3' +'_dilated_block_secondRELU')(x_more)
        x_add = Add(name=str(stage) + '_' + str(block) + '_3' + '_dilated_block_Add')([x_more, x_shortcut])
        x_dilated_output = Activation('relu', name=str(stage)+'_' + str(block) +'_dilated_block_relu')(x_add)
        return x_dilated_output

    def dilated_with_projection(self, inputs, stage):
        """
        Dilated block with 1x1 convolution projection for the shortcut, structure like res-conv-block
        """
        x_shortcut = inputs
        x = Conv2D(filters=256, kernel_size=(1, 1), padding='same', kernel_regularizer=l2(self.l2r),
                   name=str(stage) + '_1'+ '_dilated_project_first1x1')(inputs)
        x = BatchNormalization(name=str(stage) + '_1'+ '_dilated_project_firstBN')(x)
        x = Activation('relu', name=str(stage) + '_1''_dilated_project_firstRELU')(x)

        x_dilated = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                           name=str(stage) + '_2'+'_dilated_project_dilatedconv',
                           kernel_regularizer=l2(self.l2r), dilation_rate=(2, 2))(x)
        x_dilated = BatchNormalization(name=str(stage)+ '_2' + '_dilated_project_DBN')(x_dilated)
        x_dilated = Activation('relu', name=str(stage) + '_2'+ '_dialated_project_DRELU')(x_dilated)

        x_more = Conv2D(filters=256, kernel_size=(1, 1), padding='same', kernel_regularizer=l2(self.l2r),
                        name=str(stage) + '_3'+ '_dilated_project_second1x1')(x_dilated)
        x_more = BatchNormalization(name=str(stage) + '_3'+ '_dilated_project_secondBN')(x_more)
        x_more = Activation('relu',name=str(stage) + '_3'+ '_dilated_project_secondRELU')(x_more)

        x_shortcut_project = Conv2D(filters=256, kernel_size=(1, 1), padding='same', kernel_regularizer=l2(self.l2r),
                   name=str(stage) + '_dialted_project_shortcutConv')(x_shortcut)
        x_shortcut_project = BatchNormalization(name=str(stage) + '_dialted_project_shortcutBN')(x_shortcut_project)

        x_add = Add(name=str(stage) + '_dilated_project_add')([x_more, x_shortcut_project])
        x_dilated_output = Activation('relu', name=str(stage) + '_dilated_project_finalRELU')(x_add)
        return x_dilated_output

    ####################################
    # Full structure of encoder-decoder
    ####################################
    def detnet_resnet101_backbone(self):
        img_input = Input(self.input_shape)
        #########
        # Adapted first stage
        #########
        x_stage1 = self.first_layer(inputs=img_input)
        x_stage2, x_stage3, x_stage4 = self.resnet_101(x_stage1, [32, 64, 128], stages=[2, 3, 4])

        #########
        # following layer proposed by DetNet
        #########
        x_stage5_B = self.dilated_with_projection(x_stage4, stage=5)
        x_stage5_A1 = self.dilated_bottleneck(x_stage5_B, stage=5, block=1)
        x_stage5 = self.dilated_bottleneck(x_stage5_A1, stage=5, block=2)
        x_stage6_B = self.dilated_with_projection(x_stage5, stage=6)
        x_stage6_A1 = self.dilated_bottleneck(x_stage6_B, stage=6, block=1)
        x_stage6 = self.dilated_bottleneck(x_stage6_A1, stage=6, block=2)
        x_stage2_1x1 = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                              name='stage2_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage2)
        x_stage2_1x1 = BatchNormalization(name='x_stage2_1x1_BN')(x_stage2_1x1)
        x_stage3_1x1 = Conv2D(filters=2, kernel_size=(1,1), padding='same',
                              name = 'stage3_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage3)
        x_stage3_1x1 = BatchNormalization(name='x_stage3_1x1_BN')(x_stage3_1x1)
        x_stage4_1x1 = Conv2D(filters=2, kernel_size=(1,1), padding='same',
                              name = 'stage4_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage4)
        x_stage4_1x1 = BatchNormalization(name='x_stage4_1x1_BN')(x_stage4_1x1)
        x_stage5_1x1 = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                              name='stage5_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage5)
        x_stage5_1x1 = BatchNormalization(name='x_stage5_1x1_BN')(x_stage5_1x1)
        x_stage6_1x1 = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                              name='stage6_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage6)
        x_stage6_1x1 = BatchNormalization(name='x_stage6_1x1_BN')(x_stage6_1x1)

        stage_56 = Add(name='stage5_add_6')([x_stage6_1x1, x_stage5_1x1])
        stage_456 = Add(name='stage4_add_56')([stage_56, x_stage4_1x1])
        stage_456_upsample = Conv2DTranspose(filters=2, kernel_size=(1, 1), strides=(2, 2),
                                             kernel_regularizer=keras.regularizers.l2(self.l2r),
                                             name='stage456_upsample_Deconv')(stage_456)
        stage_456_upsample = BatchNormalization(name='stage_456_upsample_BN')(stage_456_upsample)


        stage_3456 = Add(name='stage3_add_456')([stage_456_upsample, x_stage3_1x1])

        stage_3456_upsample = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2),
                                              padding='same',
                                              kernel_regularizer=keras.regularizers.l2(self.l2r),
                                              name='stage3456_upsample')(stage_3456)
        stage_3456_upsample = BatchNormalization(name='stage_3456_upsample_BN')(stage_3456_upsample)

        stage_23456 = Add(name='stage2_add_3456')([stage_3456_upsample, x_stage2_1x1])
        x_output_b4_softmax = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2),
                                              padding='same', kernel_regularizer=l2(self.l2r),
                                              name='Deconv_b4_softmax_output')(stage_23456)
        x_output_b4_softmax = BatchNormalization(name='x_output_b4_softmax_BN')(x_output_b4_softmax)

        x_output = Activation('softmax', name='Final_Softmax')(x_output_b4_softmax)
        detnet_model = Model(inputs=img_input,
                             outputs=x_output)
        return detnet_model

    def detnet_resnet50_backbone(self):
        img_input = Input(self.input_shape)
        #########
        # Adapted first stage
        #########
        x_stage1 = self.first_layer(inputs=img_input)
        x_stage2, x_stage3, x_stage4 = self.resnet_50(x_stage1, [32, 64, 128], stages=[2, 3, 4])

        #########
        # following layer proposed by DetNet
        #########
        x_stage5_B = self.dilated_with_projection(x_stage4, stage=5)
        x_stage5_A1 = self.dilated_bottleneck(x_stage5_B, stage=5, block=1)
        x_stage5 = self.dilated_bottleneck(x_stage5_A1, stage=5, block=2)
        x_stage6_B = self.dilated_with_projection(x_stage5, stage=6)
        x_stage6_A1 = self.dilated_bottleneck(x_stage6_B, stage=6, block=1)
        x_stage6 = self.dilated_bottleneck(x_stage6_A1, stage=6, block=2)
        x_stage2_1x1 = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                              name='stage2_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage2)
        x_stage3_1x1 = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                              name='stage3_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage3)
        x_stage4_1x1 = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                              name='stage4_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage4)
        x_stage5_1x1 = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                              name='stage5_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage5)
        x_stage6_1x1 = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                              name='stage6_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage6)

        stage_56 = Add(name='stage5_add_6')([x_stage6_1x1, x_stage5_1x1])
        stage_456 = Add(name='stage4_add_56')([stage_56, x_stage4_1x1])
        stage_456_upsample = Conv2DTranspose(filters=2, kernel_size=(1, 1), strides=(2, 2),
                                             kernel_regularizer=keras.regularizers.l2(self.l2r),
                                             name='stage456_upsample')(stage_456)
        stage_3456 = Add(name='stage3_add_456')([stage_456_upsample, x_stage3_1x1])

        stage_3456_upsample = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2),
                                              padding='same',
                                              kernel_regularizer=keras.regularizers.l2(self.l2r),
                                              name='stage3456_upsample')(stage_3456)
        stage_23456 = Add(name='stage2_add_3456')([stage_3456_upsample, x_stage2_1x1])
        x_output_b4_softmax = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2),
                                              padding='same', kernel_regularizer=l2(self.l2r),
                                              name='Deconv_b4_softmax_output')(stage_23456)
        x_output = Activation('softmax', name='Final_Softmax')(x_output_b4_softmax)
        detnet_model = Model(inputs=img_input,
                             outputs=x_output)
        return detnet_model

    def detnet_resnet152_backbone(self):
        img_input = Input(self.input_shape)
        #########
        # Adapted first stage
        #########
        x_stage1 = self.first_layer(inputs=img_input)
        x_stage2, x_stage3, x_stage4 = self.resnet_152(x_stage1, [32, 64, 128], stages=[2, 3, 4])

        #########
        # following layer proposed by DetNet
        #########
        x_stage5_B = self.dilated_with_projection(x_stage4, stage=5)
        x_stage5_A1 = self.dilated_bottleneck(x_stage5_B, stage=5, block=1)
        x_stage5 = self.dilated_bottleneck(x_stage5_A1, stage=5, block=2)
        x_stage6_B = self.dilated_with_projection(x_stage5, stage=6)
        x_stage6_A1 = self.dilated_bottleneck(x_stage6_B, stage=6, block=1)
        x_stage6 = self.dilated_bottleneck(x_stage6_A1, stage=6, block=2)
        x_stage2_1x1 = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                              name='stage2_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage2)
        x_stage3_1x1 = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                              name='stage3_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage3)
        x_stage4_1x1 = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                              name='stage4_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage4)
        x_stage5_1x1 = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                              name='stage5_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage5)
        x_stage6_1x1 = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                              name='stage6_1x1_conv',
                              kernel_regularizer=l2(self.l2r))(x_stage6)

        stage_56 = Add(name='stage5_add_6')([x_stage6_1x1, x_stage5_1x1])
        stage_456 = Add(name='stage4_add_56')([stage_56, x_stage4_1x1])
        stage_456_upsample = Conv2DTranspose(filters=2, kernel_size=(1, 1), strides=(2, 2),
                                             kernel_regularizer=keras.regularizers.l2(self.l2r),
                                             name='stage456_upsample')(stage_456)
        stage_3456 = Add(name='stage3_add_456')([stage_456_upsample, x_stage3_1x1])

        stage_3456_upsample = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2),
                                              padding='same',
                                              kernel_regularizer=keras.regularizers.l2(self.l2r),
                                              name='stage3456_upsample')(stage_3456)
        stage_23456 = Add(name='stage2_add_3456')([stage_3456_upsample, x_stage2_1x1])
        x_output_b4_softmax = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2),
                                              padding='same', kernel_regularizer=l2(self.l2r),
                                              name='Deconv_b4_softmax_output')(stage_23456)
        x_output = Activation('softmax', name='Final_Softmax')(x_output_b4_softmax)
        detnet_model = Model(inputs=img_input,
                             outputs=x_output)
        return detnet_model
    """
    def detnet_decoder(self, features):
        stage_56 = Add(name='stage5_add_6')([features[3], features[4]])
        stage_456 = Add(name='stage4_add_56')([stage_56, features[2]])
        stage_456_upsample = Conv2DTranspose(filters=2, kernel_size=(1,1), strides=(2,2),
                                             kernel_regularizer=keras.regularizers.l2(self.l2r),
                                             name='stage456_upsample')(stage_456)
        stage_3456 = Add(name='stage3_add_456')([features[1], stage_456_upsample])

        stage_3456_upsample = Conv2DTranspose(filters=2, kernel_size=(3,3), strides=(2,2),
                                              padding='same',
                                              kernel_regularizer=keras.regularizers.l2(self.l2r),
                                              name='stage3456_upsample')(stage_3456)
        stage_23456 = Add(name='stage2_add_3456')([features[0], stage_3456_upsample])
        return stage_23456
   
    def detnet_backbone(self, softmax_trainable=True):
        img_input = Input(self.input_shape)
        x_encoder = self.detnet_encoder(img_input)
        x_decoder = self.detnet_decoder(x_encoder)
        x_output_b4_softmax = Conv2DTranspose(filters=2, kernel_size=(3,3), strides=(2,2),
                                             padding='same', kernel_regularizer=l2(self.l2r),
                                             name='Deconv_b4_softmax_output')(x_decoder)
        x_output = Activation('softmax', name='Final_Softmax')(x_output_b4_softmax)
        detnet_model = Model(inputs=img_input,
                             outputs=x_output)
        return detnet_model
    """


class TimerCallback(Callback):
    """Tracking time spend on each epoch as well as whole training process.
    """
    def __init__(self):
        super(TimerCallback, self).__init__()
        self.epoch_time = 0
        self.training_time = 0

    def on_train_begin(self, logs=None):
        self.training_time = time.time()

    def on_train_end(self, logs=None):
        end_time = np.round(time.time() - self.training_time, 2)
        time_to_min = end_time / 60
        print('training takes {} minutes'.format(time_to_min))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print('epoch takes {} seconds to train'.format(np.round(time.time() - self.epoch_time), 2))


def data_prepare(print_image_shape=False, print_input_shape=False):
    """
    prepare data for model.
    :param print_image_shape: print image shape if set true.
    :param print_input_shape: print input shape(after categorize) if set true
    :return: list of input to model
    """
    def reshape_mask(origin, cate, num_class):
        return cate.reshape((origin.shape[0], origin.shape[1], origin.shape[2], num_class))

    train_imgs, train_det_masks, train_cls_masks = load_data(data_path=DATA_DIR, type='train',
                                                             reshape_size=(512, 512))
    valid_imgs, valid_det_masks, valid_cls_masks = load_data(data_path=DATA_DIR, type='validation',
                                                             reshape_size=(512, 512))
    test_imgs, test_det_masks, test_cls_masks = load_data(data_path=DATA_DIR, type='test',
                                                          reshape_size=(512, 512))

    if print_image_shape:
        print('Image shape print below: ')
        print('train_imgs: {}, train_det_masks: {}'.format(train_imgs.shape, train_det_masks.shape))
        print('valid_imgs: {}, valid_det_masks: {}'.format(valid_imgs.shape, valid_det_masks.shape))
        print('test_imgs: {}, test_det_masks: {}'.format(test_imgs.shape, test_det_masks.shape))
        print()

    train_det = np_utils.to_categorical(train_det_masks, 2)
    train_det = reshape_mask(train_det_masks, train_det, 2)

    valid_det = np_utils.to_categorical(valid_det_masks, 2)
    valid_det = reshape_mask(valid_det_masks, valid_det, 2)

    test_det = np_utils.to_categorical(test_det_masks, 2)
    test_det = reshape_mask(test_det_masks, test_det, 2)

    if print_input_shape:
        print('input shape print below: ')
        print('train_imgs: {}, train_det: {}'.format(train_imgs.shape, train_det.shape))
        print('valid_imgs: {}, valid_det: {}'.format(valid_imgs.shape, valid_det.shape))
        print('test_imgs: {}, test_det: {}'.format(test_imgs.shape, test_det.shape))
        print()
    return [train_imgs, train_det,valid_imgs, valid_det,  test_imgs, test_det,]


def aug_on_fly(img, det_mask, cls_mask):
    """Do augmentation with different combination on each training batch
    """
    def image_basic_augmentation(image, masks, ratio_operations=0.9):
        # without additional operations
        # according to the paper, operations such as shearing, fliping horizontal/vertical,
        # rotating, zooming and channel shifting will be apply
        sometimes = lambda aug: iaa.Sometimes(ratio_operations, aug)
        hor_flip_angle = np.random.uniform(0, 1)
        ver_flip_angle = np.random.uniform(0, 1)
        seq = iaa.Sequential([
            sometimes(
                iaa.SomeOf((0, 5), [
                iaa.Fliplr(hor_flip_angle),
                iaa.Flipud(ver_flip_angle),
                iaa.Affine(shear=(-16, 16)),
                iaa.Affine(scale={'x': (1, 1.6), 'y': (1, 1.6)}),
                iaa.PerspectiveTransform(scale=(0.01, 0.1))
            ]))
        ])
        det_mask, cls_mask = masks[0], masks[1]
        seq_to_deterministic = seq.to_deterministic()
        aug_img = seq_to_deterministic.augment_images(image)
        aug_det_mask = seq_to_deterministic.augment_images(det_mask)
        aug_cls_mask = seq_to_deterministic.augment_images(cls_mask)
        return aug_img, aug_det_mask, aug_cls_mask

    aug_image, aug_det_mask, aug_cls_mask = image_basic_augmentation(image=img, masks=[det_mask, cls_mask])
    return aug_image, aug_det_mask, aug_cls_mask

def heavy_aug_on_fly(img, det_mask):
    """Do augmentation with different combination on each training batch
    """

    def image_heavy_augmentation(image, det_masks, ratio_operations=0.6):
        # according to the paper, operations such as shearing, fliping horizontal/vertical,
        # rotating, zooming and channel shifting will be apply
        sometimes = lambda aug: iaa.Sometimes(ratio_operations, aug)
        edge_detect_sometime = lambda aug: iaa.Sometimes(0.1, aug)
        elasitic_sometime = lambda aug:iaa.Sometimes(0.2, aug)
        add_gauss_noise = lambda aug: iaa.Sometimes(0.15, aug)
        hor_flip_angle = np.random.uniform(0, 1)
        ver_flip_angle = np.random.uniform(0, 1)
        seq = iaa.Sequential([
            iaa.SomeOf((0, 5), [
                iaa.Fliplr(hor_flip_angle),
                iaa.Flipud(ver_flip_angle),
                iaa.Affine(shear=(-16, 16)),
                iaa.Affine(scale={'x': (1, 1.6), 'y': (1, 1.6)}),
                iaa.PerspectiveTransform(scale=(0.01, 0.1)),

                # These are additional augmentation.
                #iaa.ContrastNormalization((0.75, 1.5))

            ])
            #)
        ])#, random_order=True)
        """
                    edge_detect_sometime(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.7)),
                        iaa.DirectedEdgeDetect(alpha=(0,0.7), direction=(0.0, 1.0)
                                               )
                    ])),
                    add_gauss_noise(iaa.AdditiveGaussianNoise(loc=0,
                                                              scale=(0.0, 0.05*255),
                                                              per_channel=0.5)
                                    ),
                    iaa.Sometimes(0.3,
                                  iaa.GaussianBlur(sigma=(0, 0.5))
                                  ),
                    elasitic_sometime(
                        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                    """
        seq_to_deterministic = seq.to_deterministic()
        aug_img = seq_to_deterministic.augment_images(image)
        aug_det_mask = seq_to_deterministic.augment_images(det_masks)
        return aug_img, aug_det_mask

    aug_image, aug_det_mask = image_heavy_augmentation(image=img, det_masks=det_mask)
    return aug_image, aug_det_mask


def ori_shape_generator_with_heavy_aug(features, det_labels, batch_size,
                                       aug_num=25):
    batch_features = np.zeros((batch_size * aug_num, 512, 512, 3))
    batch_det_labels = np.zeros((batch_size * aug_num, 512, 512, 2))
    while True:
        counter = 0
        for i in range(batch_size):
            index = np.random.choice(features.shape[0], 1)
            feature_index = features[index]
            det_label_index = det_labels[index]

            for k in range(aug_num):
                aug_feature, aug_det_label= heavy_aug_on_fly(feature_index, det_label_index)
                batch_features[counter] = aug_feature
                batch_det_labels[counter] = aug_det_label
                counter = counter + 1

        yield batch_features, batch_det_labels


def detnet_model_compile(nn, det_loss_weight,
                         optimizer, summary=False,
                         fkg_smooth_factor=None,
                         bkg_smooth_factor=None,
                         ind_factor=None):
    """

    :param det_loss_weight:
    :param kernel_weight:
    :param summary:
    :return:
    """
    # Use focal loss if model_loss set to focal, otherwise basic crossentropy.
    if Config.model_loss == 'focal':
        loss_input = detection_focal_loss_K(det_loss_weight, fkg_smooth_factor)
    elif Config.model_loss == 'base':
        loss_input = detection_loss_K(det_loss_weight)
    elif Config.model_loss == 'base_no':
        loss_input = detection_loss_without_part2_K(det_loss_weight)
    elif Config.model_loss == 'focal_no':
        loss_input = detection_focal_loss_without_part2_K(det_loss_weight, fkg_smooth_factor)
    elif Config.model_loss == 'focal_double':
        loss_input = detection_double_focal_loss_K(det_loss_weight,
                                                   fkg_smooth_factor,
                                                   bkg_smooth_factor)
    elif Config.model_loss == 'focal_double_ind':
        loss_input = detection_double_focal_loss_indicator_K(det_loss_weight, fkg_smooth_factor,
                                                             bkg_smooth_factor, indicator_weight=ind_factor)
    elif Config.model_loss == 'default':
        loss_input = ['categorical_crossentropy']


    print('detection model is set')
    if Config.backbone == 'resnet50':
        detnet_model=nn.detnet_resnet50_backbone()
    elif Config.backbone == 'resnet101':
        detnet_model=nn.detnet_resnet101_backbone()
    elif Config.backbone == 'resnet152':
        detnet_model=nn.detnet_resnet152_backbone()
    print('The backbone structure is using {}'.format(Config.backbone))
    detnet_model.compile(optimizer=optimizer,
                      loss=loss_input,
                      metrics=['accuracy'])
    if summary==True:
        detnet_model.summary()
    return detnet_model


def callback_preparation(model, hyper):
    """
    implement necessary callbacks into model.
    :return: list of callback.
    """
    timer = TimerCallback()
    timer.set_model(model)
    tensorboard_callback = TensorBoard(os.path.join(TENSORBOARD_DIR, hyper +'_tb_logs'))
    checkpoint_callback = ModelCheckpoint(os.path.join(CHECKPOINT_DIR,
                                                       hyper + '_cp.h5'), period=1)
    return [tensorboard_callback, checkpoint_callback, timer]


def tune_loss_weight():
    """
    use this function to fine tune weights later.
    :return:
    """
    print('weight initialized')
    det_weight = [np.array([0.2, 0.8]),np.array([0.1, 0.9]), np.array([0.15, 0.85])]
    l2_weight = 0.001

    fkg_smooth_factor = [0.5, 1, 2, 3, 4, 5]
    bkg_smooth_factor = [0.5, 1, 2, 3, 4, 5]
    fkb_extend_smooth_factor = [0.6, 0.7, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.1, 3.2, 3.3,
                                3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.1, 4.2, 4.3, 4.4, 4.5]
    bkg_extend_smooth_factor = [0.6, 0.7, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.1, 3.2, 3.3,
                                3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.1, 4.2, 4.3, 4.4, 4.5]
    det_extend_weight = [np.array([0.21, 0.79]), np.array([0.19, 0.81]),
                         np.array([0.18, 0.82]), np.array([0.17, 0.83]),
                         np.array([0.22, 0.78]), np.array([0.23, 0.77]),
                         np.array([0.24, 0.76])]
    ind_factor = [np.array([0.2, 0.8]),np.array([0.1, 0.9]), np.array([0.15, 0.85])]
    return [det_weight, fkg_smooth_factor, l2_weight, bkg_smooth_factor,
            fkb_extend_smooth_factor, bkg_extend_smooth_factor, det_extend_weight, ind_factor]


def save_model_weights(hyper):
    """
    Set the path to save model weights after training finishes.
    :param hyper:
    :return:
    """
    det_model_weights_saver = os.path.join(WEIGHTS_DIR, str(Config.model_loss) + '_' + hyper + '_train.h5')

    return det_model_weights_saver


def lr_scheduler(epoch):
    lr = 0.01
    if epoch < 100 and epoch != 0:
        lr = lr - 0.0001
    if epoch % 10 == 0:
        print('Current learning rate is :{}'.format(lr))
    if epoch == 100:
        lr = 0.001
        print('Learning rate is modified after 100 epoch {}'.format(lr))
    if epoch == 150:
        lr = 0.0001
    if epoch == 200:
        lr = 0.00001
    if epoch == 250:
        lr = 0.000001
    return lr


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.gpu)
    hyper_para = tune_loss_weight()
    CROP_SIZE = 64
    BATCH_SIZE = Config.image_per_gpu * Config.gpu_count

    EPOCHS = 250


    if Config.backbone == 'resnet101':
        NUM_TO_AUG = 6
        TRAIN_STEP_PER_EPOCH = 32
    elif Config.backbone == 'resnet152':
        NUM_TO_AUG = 5
        TRAIN_STEP_PER_EPOCH = 40
    elif Config.backbone == 'resnet50':
        NUM_TO_AUG = 5
        TRAIN_STEP_PER_EPOCH = 40
    #NUM_TO_CROP, NUM_TO_AUG = 20, 10

    data = data_prepare(print_input_shape=True, print_image_shape=True)
    network = Detnet()
    optimizer = SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=True)

    for i, det_weight in enumerate(hyper_para[0]):
        if Config.model_loss == 'focal_double':
            print('------------------------------------')
            print('This model is using {}'.format(Config.model_loss))
            print()
            for j, fkg_weight in enumerate(hyper_para[1]):
                for k, bkg_weight in enumerate(hyper_para[3]):
                    hyper = '{}_loss:{}_det:{}_fkg:{}_bkg:{}_lr:0.01'.format(Config.backbone, Config.model_loss,
                                                                             det_weight[0],
                                                                             fkg_weight,
                                                                             bkg_weight)  # _l2:{}_bkg:{}'.format()
                    print(hyper)
                    print()
                    model_weights_saver = save_model_weights(hyper)

                    detnet_model = detnet_model_compile(nn=network,
                                                        summary=Config.summary,
                                                        det_loss_weight=det_weight,
                                                        optimizer=optimizer,
                                                        fkg_smooth_factor=fkg_weight,
                                                        bkg_smooth_factor=bkg_weight)
                    print('base detection is training')
                    list_callback = callback_preparation(detnet_model, hyper)
                    list_callback.append(LearningRateScheduler(lr_scheduler))
                    detnet_model.fit_generator(ori_shape_generator_with_heavy_aug(data[0],
                                                                                  data[1],
                                                                                  batch_size=BATCH_SIZE,
                                                                                  aug_num=NUM_TO_AUG),
                                               epochs=EPOCHS,
                                               steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                                               validation_data=ori_shape_generator_with_heavy_aug(
                                                   data[2], data[3], batch_size=BATCH_SIZE,
                                                   aug_num=NUM_TO_AUG),
                                               validation_steps=2,
                                               callbacks=list_callback)

                    detnet_model.save_weights(model_weights_saver)

        if Config.model_loss == 'focal' or Config.model_loss == 'focal_no':
            print('------------------------------------')
            print('This model is using {}'.format(Config.model_loss))
            for order, loss in enumerate(hyper_para[1]):
                hyper = '{}_loss:{}_det:{}_lr:0.01'.format(Config.backbone, Config.model_loss, loss, det_weight[0])
                print(hyper)
                print()
                model_weights_saver = save_model_weights(hyper)

                detnet_model = detnet_model_compile(nn=network,
                                                    summary=Config.summary,
                                                    det_loss_weight=det_weight,
                                                    optimizer=optimizer,
                                                    fkg_smooth_factor=loss)
                print('base detection is training')
                list_callback = callback_preparation(detnet_model, hyper)
                list_callback.append(LearningRateScheduler(lr_scheduler))
                detnet_model.fit_generator(ori_shape_generator_with_heavy_aug(data[0],
                                                                              data[1],
                                                                              batch_size=BATCH_SIZE,
                                                                              aug_num=NUM_TO_AUG),
                                        epochs=EPOCHS,
                                        steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                                        validation_data=ori_shape_generator_with_heavy_aug(
                                                            data[2], data[3], batch_size=BATCH_SIZE,
                                                            aug_num=NUM_TO_AUG),
                                        validation_steps=2,
                                        callbacks=list_callback)

                detnet_model.save_weights(model_weights_saver)
        if Config.model_loss == 'base' or Config.model_loss == 'base_no':
            print('------------------------------------')
            print('This model is using {}'.format(Config.model_loss))
            hyper = '{}_loss:{}_lr:0.01_bkg:{}_'.format(Config.backbone, Config.model_loss, det_weight[0])
            print(hyper)
            print()
            model_weights_saver = save_model_weights(hyper)

            detnet_model = detnet_model_compile(nn=network,
                                                summary=Config.summary,
                                                det_loss_weight=det_weight,
                                                optimizer=optimizer)
            print('base detection is training')
            list_callback = callback_preparation(detnet_model, hyper)
            list_callback.append(LearningRateScheduler(lr_scheduler))
            detnet_model.fit_generator(ori_shape_generator_with_heavy_aug(data[0],
                                                                          data[1],
                                                                          batch_size=BATCH_SIZE,
                                                                          aug_num=NUM_TO_AUG),
                                       epochs=EPOCHS,
                                       steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                                       validation_data=ori_shape_generator_with_heavy_aug(
                                           data[2], data[3], batch_size=BATCH_SIZE,
                                           aug_num=NUM_TO_AUG),
                                       validation_steps=2,
                                       callbacks=list_callback)

            detnet_model.save_weights(model_weights_saver)


    # Extension program.
    if Config.model_loss == 'focal_double' or Config.extend_program == True:
        for i, det_weight in enumerate(hyper_para[6]):
            if Config.model_loss == 'focal_double':
                print("--------------------------------------")
                print('This model is using {}'.format(Config.model_loss))
                for j, fkg_weight in enumerate(hyper_para[4]):
                    for k, bkg_weight in enumerate(hyper_para[5]):
                        hyper = '{}_loss:{}_det:{}_fkg:{}_bkg:{}_lr:0.01'.format(Config.backbone,
                                                                                 Config.model_loss,
                                                                                 det_weight[0],
                                                                                 fkg_weight,
                                                                                 bkg_weight)
                        print(hyper)
                        print()
                        model_weights_saver = save_model_weights(hyper)

                        detnet_model = detnet_model_compile(nn=network,
                                                            summary=Config.summary,
                                                            det_loss_weight=det_weight,
                                                            optimizer=optimizer,
                                                            fkg_smooth_factor=fkg_weight,
                                                            bkg_smooth_factor=bkg_weight)
                        print('base detection is training')
                        list_callback = callback_preparation(detnet_model, hyper)
                        list_callback.append(LearningRateScheduler(lr_scheduler))
                        detnet_model.fit_generator(ori_shape_generator_with_heavy_aug(data[0],
                                                                                      data[1],
                                                                                      batch_size=BATCH_SIZE,
                                                                                      aug_num=NUM_TO_AUG),
                                                   epochs=EPOCHS,
                                                   steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                                                   validation_data=ori_shape_generator_with_heavy_aug(
                                                       data[2], data[3], batch_size=BATCH_SIZE,
                                                       aug_num=NUM_TO_AUG),
                                                   validation_steps=2,
                                                   callbacks=list_callback)

                        detnet_model.save_weights(model_weights_saver)



