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
from util import load_data, set_gpu, set_num_step_and_aug, lr_scheduler, aug_on_fly, heavy_aug_on_fly
import os, time
from image_augmentation import ImageCropping
from loss import detection_focal_loss_K, detection_loss_K, detection_double_focal_loss_K, detection_double_focal_loss_indicator_K
from config import Config
from tensorflow.python.client import device_lib
#from encoder_decoder_object_det import Conv3l2

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
###########################################
# ResNet Graph
###########################################
def identity_block_2(f, stage, block, inputs, l2_weight, trainable=True):
    """
    :param f: number of filters
    :param stage: stage of residual blocks
    :param block: ith module
    :param trainable: freeze layer if false
    """
    x_shortcut = inputs

    x = Conv3l2(filters=f, kernel_regularizer_weight=l2_weight,
                name=str(stage) + '_' + str(block) + '_idblock_conv_1',
                trainable=trainable)(inputs)
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_idblock_BN_1',
                           trainable=trainable)(x)
    x = Activation('relu', name=str(stage) + '_' + str(block) + '_idblock_act_1',
                   trainable=trainable)(x)

    x = Conv3l2(filters=f, kernel_regularizer_weight=l2_weight,
                name=str(stage) + '_' + str(block) + '_idblock_conv_2')(x)
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_idblock_BN_2',
                           trainable=trainable)(x)

    x_add = Add(name=str(stage) + '_' + str(block) + '_idblock_add', trainable=trainable)([x, x_shortcut])
    x_idblock_output = Activation('relu', name=str(stage) + '_' + str(block) + '_idblock_act_outout',
                                  trainable=trainable)(x_add)
    return x_idblock_output


def convolution_block_2(f, stage, block, inputs, l2_weight, trainable=True):
    """
    :param f: number of filters
    :param stage: stage of residual blocks
    :param block: ith module
    """
    x = Conv3l2(filters=f, strides=(2, 2), kernel_regularizer_weight=l2_weight,
                name=str(stage) + str(block) + '_' + '_convblock_conv_1',
                trainable=trainable)(inputs)
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_convblock_BN_1',
                           trainable=trainable)(x)
    x = Activation('relu', name=str(stage) + '_' + str(block) + '_convblock_act_1',
                   trainable=trainable)(x)
    x = Conv3l2(filters=f, kernel_regularizer_weight=l2_weight,
                name=str(stage) + '_' + str(block) + '_convblock_conv_2',
                trainable=trainable)(x)
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_convblock_BN_2',
                           trainable=trainable)(x)


    x_shortcut = Conv2D(f, kernel_size=(1, 1), strides=(2, 2), padding='same',
                        kernel_regularizer=keras.regularizers.l2(l2_weight),
                        name=str(stage) + '_' + str(block) + '_convblock_shortcut_conv',
                        trainable=trainable)(inputs)
    x_shortcut = BatchNormalization(name=str(stage) + '_' + str(block) + '_convblock_shortcut_BN_1',
                                    trainable=trainable)(x_shortcut)
    x_add = Add(name=str(stage) + '_' + str(block) + '_convblock_add',
                trainable=trainable)([x, x_shortcut])
    x_convblock_output = Activation('relu', name=str(stage) + '_' + str(block) + '_convblock_act_output',
                                    trainable=trainable)(x_add)
    return x_convblock_output

def dilated_bottleneck(inputs, stage, block, l2_weight):
    """
    Dilated block without 1x1 convolution projection, structure like res-id-block
    """
    x_shortcut = inputs
    x = Conv2D(filters=256, kernel_size=(1,1), padding='same', kernel_regularizer=l2(l2_weight),
               name=str(stage) + '_' + str(block) + '_1' + '_dilated_block_first1x1')(inputs)
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_1'+ '_dilated_block_firstBN')(x)
    x = Activation('relu', name=str(stage) + '_' + str(block) + '_1'+ '_dilated_block_firstRELU')(x)

    x_dilated = Conv2D(filters=256, kernel_size=(3,3), padding='same',
                       name=str(stage) + '_' + str(block) + '_2' + '_dilated_block_dilatedconv',
                       kernel_regularizer=l2(l2_weight), dilation_rate=(2,2))(x)
    x_dilated = BatchNormalization(name=str(stage) + '_'+ str(block) + '_2'+ '_dilated_block_dilatedBN')(x_dilated)
    x_dilated = Activation('relu',name=str(stage) +'_'+ str(block) + '_2'+ '_dilated_block_dilatedRELU')(x_dilated)

    x_more = Conv2D(filters=256, kernel_size=(1,1), padding='same', kernel_regularizer=l2(l2_weight),
                    name=str(stage) + '_'+ str(block) + '_3'+ '_dilated_block_second1x1')(x_dilated)
    x_more = BatchNormalization(name=str(stage) + '_' + str(block) + '_3'+ '_dilated_block_secondBN')(x_more)
    x_more = Activation('relu', name=str(stage) + str(block) + '_3' +'_dilated_block_secondRELU')(x_more)
    x_add = Add(name=str(stage) + '_' + str(block) + '_3' + '_dilated_block_Add')([x_more, x_shortcut])
    x_dilated_output = Activation('relu', name=str(stage)+'_' + str(block) +'_dilated_block_relu')(x_add)
    return x_dilated_output

def dilated_with_projection(inputs, stage, l2_weight):
    """
    Dilated block with 1x1 convolution projection for the shortcut, structure like res-conv-block
    """
    x_shortcut = inputs
    x = Conv2D(filters=256, kernel_size=(1, 1), padding='same', kernel_regularizer=l2(l2_weight),
               name=str(stage) + '_1'+ '_dilated_project_first1x1')(inputs)
    x = BatchNormalization(name=str(stage) + '_1'+ '_dilated_project_firstBN')(x)
    x = Activation('relu', name=str(stage) + '_1''_dilated_project_firstRELU')(x)

    x_dilated = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                       name=str(stage) + '_2'+'_dilated_project_dilatedconv',
                       kernel_regularizer=l2(l2_weight), dilation_rate=(2, 2))(x)
    x_dilated = BatchNormalization(name=str(stage)+ '_2' + '_dilated_project_DBN')(x_dilated)
    x_dilated = Activation('relu', name=str(stage) + '_2'+ '_dialated_project_DRELU')(x_dilated)

    x_more = Conv2D(filters=256, kernel_size=(1, 1), padding='same', kernel_regularizer=l2(l2_weight),
                    name=str(stage) + '_3'+ '_dilated_project_second1x1')(x_dilated)
    x_more = BatchNormalization(name=str(stage) + '_3'+ '_dilated_project_secondBN')(x_more)
    x_more = Activation('relu',name=str(stage) + '_3'+ '_dilated_project_secondRELU')(x_more)

    x_shortcut_project = Conv2D(filters=256, kernel_size=(1, 1), padding='same', kernel_regularizer=l2(l2_weight),
               name=str(stage) + '_dialted_project_shortcutConv')(x_shortcut)
    x_shortcut_project = BatchNormalization(name=str(stage) + '_dialted_project_shortcutBN')(x_shortcut_project)

    x_add = Add(name=str(stage) + '_dilated_project_add')([x_more, x_shortcut_project])
    x_dilated_output = Activation('relu', name=str(stage) + '_dilated_project_finalRELU')(x_add)
    return x_dilated_output


def fcn_27(inputs, l2_weight, stages=[2, 3, 4, 5], filters=[32, 64, 128], trainable=True):
    x = inputs
    x_id_20 = identity_block_2(f=filters[0], stage=stages[0], block=1, inputs=x, trainable=trainable, l2_weight=l2_weight)
    x_id_21 = identity_block_2(f=filters[0], stage=stages[0], block=2, inputs=x_id_20, trainable=trainable, l2_weight=l2_weight)
    x_id_22 = identity_block_2(f=filters[0], stage=stages[0], block=3, inputs=x_id_21, trainable=trainable, l2_weight=l2_weight)
    x_id_23 = identity_block_2(f=filters[0], stage=stages[0], block=4, inputs=x_id_22, trainable=trainable, l2_weight=l2_weight)
    x_id_24 = identity_block_2(f=filters[0], stage=stages[0], block=5, inputs=x_id_23, trainable=trainable, l2_weight=l2_weight)
    x_id_25 = identity_block_2(f=filters[0], stage=stages[0], block=6, inputs=x_id_24, trainable=trainable, l2_weight=l2_weight)
    x_id_26 = identity_block_2(f=filters[0], stage=stages[0], block=7, inputs=x_id_25, trainable=trainable, l2_weight=l2_weight)
    x_id_27 = identity_block_2(f=filters[0], stage=stages[0], block=8, inputs=x_id_26, trainable=trainable, l2_weight=l2_weight)
    x_id_28 = identity_block_2(f=filters[0], stage=stages[0], block=9, inputs=x_id_27, trainable=trainable, l2_weight=l2_weight)

    x_conv_3 = convolution_block_2(f=filters[1], stage=stages[1], block=1, inputs=x_id_28, trainable=trainable, l2_weight=l2_weight)
    x_id_31 = identity_block_2(f=filters[1], stage=stages[1], block=2, inputs=x_conv_3, trainable=trainable, l2_weight=l2_weight)
    x_id_32 = identity_block_2(f=filters[1], stage=stages[1], block=3, inputs=x_id_31, trainable=trainable, l2_weight=l2_weight)
    x_id_33 = identity_block_2(f=filters[1], stage=stages[1], block=4, inputs=x_id_32, trainable=trainable, l2_weight=l2_weight)
    x_id_34 = identity_block_2(f=filters[1], stage=stages[1], block=5, inputs=x_id_33, trainable=trainable, l2_weight=l2_weight)
    x_id_35 = identity_block_2(f=filters[1], stage=stages[1], block=6, inputs=x_id_34, trainable=trainable, l2_weight=l2_weight)
    x_id_36 = identity_block_2(f=filters[1], stage=stages[1], block=7, inputs=x_id_35, trainable=trainable, l2_weight=l2_weight)
    x_id_37 = identity_block_2(f=filters[1], stage=stages[1], block=8, inputs=x_id_36, trainable=trainable, l2_weight=l2_weight)
    x_id_38 = identity_block_2(f=filters[1], stage=stages[1], block=9, inputs=x_id_37, trainable=trainable, l2_weight=l2_weight)


    x_conv_4 = convolution_block_2(f=filters[2], stage=stages[2], block=1, inputs=x_id_38, trainable=trainable, l2_weight=l2_weight)
    x_id_41 = identity_block_2(f=filters[2], stage=stages[2], block=2, inputs=x_conv_4, trainable=trainable, l2_weight=l2_weight)
    x_id_42 = identity_block_2(f=filters[2], stage=stages[2], block=3, inputs=x_id_41, trainable=trainable, l2_weight=l2_weight)
    x_id_43 = identity_block_2(f=filters[2], stage=stages[2], block=4, inputs=x_id_42, trainable=trainable, l2_weight=l2_weight)
    x_id_44 = identity_block_2(f=filters[2], stage=stages[2], block=5, inputs=x_id_43, trainable=trainable, l2_weight=l2_weight)
    x_id_45 = identity_block_2(f=filters[2], stage=stages[2], block=6, inputs=x_id_44, trainable=trainable, l2_weight=l2_weight)
    x_id_46 = identity_block_2(f=filters[2], stage=stages[2], block=7, inputs=x_id_45, trainable=trainable, l2_weight=l2_weight)
    x_id_47 = identity_block_2(f=filters[2], stage=stages[2], block=8, inputs=x_id_46, trainable=trainable, l2_weight=l2_weight)
    x_id_48 = identity_block_2(f=filters[2], stage=stages[2], block=9, inputs=x_id_47, trainable=trainable, l2_weight=l2_weight)
    return x_id_28, x_id_38, x_id_48

def fcn_36(inputs, l2_weight, stages=[2, 3, 4, 5],filters=[32, 64, 128, 256], trainable=True):
    x = inputs
    x_id_20 = identity_block_2(f=filters[0], stage=stages[0], block=1, inputs=x, trainable=trainable, l2_weight=l2_weight)
    x_id_21 = identity_block_2(f=filters[0], stage=stages[0], block=2, inputs=x_id_20, trainable=trainable, l2_weight=l2_weight)
    x_id_22 = identity_block_2(f=filters[0], stage=stages[0], block=3, inputs=x_id_21, trainable=trainable, l2_weight=l2_weight)
    x_id_23 = identity_block_2(f=filters[0], stage=stages[0], block=4, inputs=x_id_22, trainable=trainable, l2_weight=l2_weight)
    x_id_24 = identity_block_2(f=filters[0], stage=stages[0], block=5, inputs=x_id_23, trainable=trainable, l2_weight=l2_weight)
    x_id_25 = identity_block_2(f=filters[0], stage=stages[0], block=6, inputs=x_id_24, trainable=trainable, l2_weight=l2_weight)
    x_id_26 = identity_block_2(f=filters[0], stage=stages[0], block=7, inputs=x_id_25, trainable=trainable, l2_weight=l2_weight)
    x_id_27 = identity_block_2(f=filters[0], stage=stages[0], block=8, inputs=x_id_26, trainable=trainable, l2_weight=l2_weight)
    x_id_28 = identity_block_2(f=filters[0], stage=stages[0], block=9, inputs=x_id_27, trainable=trainable, l2_weight=l2_weight)

    x_conv_3 = convolution_block_2(f=filters[1], stage=stages[1], block=1, inputs=x_id_28, trainable=trainable, l2_weight=l2_weight)
    x_id_31 = identity_block_2(f=filters[1], stage=stages[1], block=2, inputs=x_conv_3, trainable=trainable, l2_weight=l2_weight)
    x_id_32 = identity_block_2(f=filters[1], stage=stages[1], block=3, inputs=x_id_31, trainable=trainable, l2_weight=l2_weight)
    x_id_33 = identity_block_2(f=filters[1], stage=stages[1], block=4, inputs=x_id_32, trainable=trainable, l2_weight=l2_weight)
    x_id_34 = identity_block_2(f=filters[1], stage=stages[1], block=5, inputs=x_id_33, trainable=trainable, l2_weight=l2_weight)
    x_id_35 = identity_block_2(f=filters[1], stage=stages[1], block=6, inputs=x_id_34, trainable=trainable, l2_weight=l2_weight)
    x_id_36 = identity_block_2(f=filters[1], stage=stages[1], block=7, inputs=x_id_35, trainable=trainable, l2_weight=l2_weight)
    x_id_37 = identity_block_2(f=filters[1], stage=stages[1], block=8, inputs=x_id_36, trainable=trainable, l2_weight=l2_weight)
    x_id_38 = identity_block_2(f=filters[1], stage=stages[1], block=9, inputs=x_id_37, trainable=trainable, l2_weight=l2_weight)


    x_conv_4 = convolution_block_2(f=filters[2], stage=stages[2], block=1, inputs=x_id_38, trainable=trainable, l2_weight=l2_weight)
    x_id_41 = identity_block_2(f=filters[2], stage=stages[2], block=2, inputs=x_conv_4, trainable=trainable, l2_weight=l2_weight)
    x_id_42 = identity_block_2(f=filters[2], stage=stages[2], block=3, inputs=x_id_41, trainable=trainable, l2_weight=l2_weight)
    x_id_43 = identity_block_2(f=filters[2], stage=stages[2], block=4, inputs=x_id_42, trainable=trainable, l2_weight=l2_weight)
    x_id_44 = identity_block_2(f=filters[2], stage=stages[2], block=5, inputs=x_id_43, trainable=trainable, l2_weight=l2_weight)
    x_id_45 = identity_block_2(f=filters[2], stage=stages[2], block=6, inputs=x_id_44, trainable=trainable, l2_weight=l2_weight)
    x_id_46 = identity_block_2(f=filters[2], stage=stages[2], block=7, inputs=x_id_45, trainable=trainable, l2_weight=l2_weight)
    x_id_47 = identity_block_2(f=filters[2], stage=stages[2], block=8, inputs=x_id_46, trainable=trainable, l2_weight=l2_weight)
    x_id_48 = identity_block_2(f=filters[2], stage=stages[2], block=9, inputs=x_id_47, trainable=trainable, l2_weight=l2_weight)

    x_conv_5 = convolution_block_2(f=filters[3], stage=stages[2], block=1, inputs=x_id_48, trainable=trainable,
                                   l2_weight=l2_weight)
    x_id_51 = identity_block_2(f=filters[3], stage=stages[2], block=2, inputs=x_conv_5, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_52 = identity_block_2(f=filters[3], stage=stages[2], block=3, inputs=x_id_51, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_53 = identity_block_2(f=filters[3], stage=stages[2], block=4, inputs=x_id_52, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_54 = identity_block_2(f=filters[3], stage=stages[2], block=5, inputs=x_id_53, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_55 = identity_block_2(f=filters[3], stage=stages[2], block=6, inputs=x_id_54, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_56 = identity_block_2(f=filters[3], stage=stages[2], block=7, inputs=x_id_55, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_57 = identity_block_2(f=filters[3], stage=stages[2], block=8, inputs=x_id_56, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_58 = identity_block_2(f=filters[3], stage=stages[2], block=9, inputs=x_id_57, trainable=trainable,
                               l2_weight=l2_weight)
    return x_id_28, x_id_38, x_id_48, x_id_58

def Fcn_det():
    def __init__(self, input_shape=(256, 256, 3)):
        # self.inputs = inputs
        self.input_shape = input_shape
        l2_weight = 0.001

    def first_layer(self, inputs, trainable=True):
        """
        First convolution layer.
        """
        x = Conv3l2(filters=32, name='Conv_1',
                    kernel_regularizer_weight=self.l2_weight,
                    trainable=trainable)(inputs)
        x = BatchNormalization(name='BN_1',trainable=trainable)(x)
        x = Activation('relu', name='act_1',trainable=trainable)(x)
        return x

    ####################################
    # FCN 36 as in paper sfcn-opi
    ####################################
    def fcn36_deconv_backbone(self, l2_weight):
        #######################################
        # Extra branch for every pyramid feature
        #######################################
        def _feature_concat_deconv_branch(C7=None, C6=None, C5=None, C4=None, C3=None, C2=None):
            """

            :param features: input feature is from every feature pyramid layer,
                             should've been already connect with 1x1 convolution layer.
            """

            def _32to256(input, type):
                x_deconv64 = Conv2DTranspose(kernel_size=(3, 3),
                                              filters=256, strides=(2, 2), name=type + '_deconv_64_Conv')(input)
                x_deconv64 = BatchNormalization(name=type + '_deconv_64_BN')(x_deconv64)
                x_deconv64 = Activation('relu', name=type + '_deconv_64_RELU')(x_deconv64)
                x_deconv128 = Conv2DTranspose(kernel_size=(3, 3),
                                              filters=256, strides=(2, 2), name=type + '_deconv_128_Conv')(x_deconv64)
                x_deconv128 = BatchNormalization(name=type + '_deconv_128_BN')(x_deconv128)
                x_deconv128 = Activation('relu', name=type + '_deconv_128_RELU')(x_deconv128)
                x_deconv256 = Conv2DTranspose(kernel_size=(3, 3),
                                              filters=256, strides=(2, 2), name=type + '_deconv_256_Conv')(x_deconv128)
                return x_deconv256

            # in detnet setting, C6.shape == C5.shape == C4.shape, in fcn27 this is 1/4 of origin image
            C7_deconv = _32to256(C7, 'C7')
            C6_deconv = _32to256(C6, 'C6')
            C5_deconv = _32to256(C5, 'C5')
            C4_deconv_128 = Conv2DTranspose(kernel_size=(3, 3),
                                            filters=128, strides=(2, 2), name='C4_deconv_128_Conv')(C4)
            C4_deconv_128 = BatchNormalization(name='C4_deconv_128_BN')(C4_deconv_128)
            C4_deconv_128 = Activation('relu', name='C4_deconv_128_RELU')(C4_deconv_128)
            C4_deconv_256 = Conv2DTranspose(kernel_size=(3, 3),
                                            filters=128, strides=(2, 2), name='C4_deconv_256_Conv')(C4_deconv_128)


            C3_deconv_256 = Conv2DTranspose(kernel_size=(3, 3),
                                            filters=64, strides=(2, 2), name='C3_deconv_256_Conv')(C3)

            C23456_concat = Concatenate()([C7_deconv, C6_deconv, C5_deconv, C4_deconv_256, C3_deconv_256, C2])

            return C23456_concat

        # tf.reset_default_graph()
        img_input = Input(self.input_shape)
        #########
        # Adapted first stage
        #########
        x_stage1 = self.first_layer(inputs=img_input)
        x_stage2, x_stage3, x_stage4, x_stage5 = fcn_36(x_stage1, l2_weight=l2_weight)
        # stage3 is 1/2 size
        #########
        # following layer proposed by DetNet
        #########
        x_stage6_B = dilated_with_projection(x_stage5, stage=6, l2_weight=l2_weight)
        x_stage6_A1 = dilated_bottleneck(x_stage6_B, stage=6, block=1, l2_weight=l2_weight)
        x_stage6 = dilated_bottleneck(x_stage6_A1, stage=6, block=2, l2_weight=l2_weight)
        x_stage7_B = dilated_with_projection(x_stage6, stage=7, l2_weight=l2_weight)
        x_stage7_A1 = dilated_bottleneck(x_stage7_B, stage=7, block=1, l2_weight=l2_weight)
        x_stage7 = dilated_bottleneck(x_stage7_A1, stage=7, block=2, l2_weight=l2_weight)

        ########
        # 1x1 convolutnion part
        ########
        x_stage2_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same',
                              name='stage2_1x1_conv')(x_stage2)
        x_stage3_1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                              name='stage3_1x1_conv',
                              kernel_regularizer=l2(l2_weight))(x_stage3)
        x_stage4_1x1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                              name='stage4_1x1_conv',
                              kernel_regularizer=l2(l2_weight))(x_stage4)
        # stage5 is 1/8 size, same as 6 and 7
        x_stage5_1x1 = Conv2D(filters=256, kernel_size=(1, 1), padding='same',
                              name='stage5_1x1_conv', kernel_regularizer=l2(l2_weight))(x_stage5)
        x_stage6_1x1 = Conv2D(filters=256, kernel_size=(1, 1), padding='same',
                              name='stage6_1x1_conv', kernel_regularizer=l2(l2_weight))(x_stage6)
        x_stage7_1x1 = Conv2D(filters=256, kernel_size=(1, 1), padding='same',
                              name='stage7_1x1_conv', kernel_regularizer=l2(l2_weight))(x_stage7)

        stage_67 = Add(name='add_stage_6_7')([x_stage6_1x1, x_stage7_1x1])
        stage_567 = Add(name='add_stage_5_6_7')([x_stage5_1x1, stage_67])
        stage_567_upsample = Conv2DTranspose(filters=128, kernel_size=(1, 1), strides=(2, 2),
                                             name='stage_567_upsample')(stage_567)
        stage_4567 = Add(name='add_stage_4_567')([stage_567_upsample, x_stage4_1x1])
        stage_4567_upsample = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2),
                                              padding='same',
                                              kernel_regularizer=keras.regularizers.l2(l2_weight),
                                              name='stage_4567_upsample')(stage_4567)
        stage_34567 = Add(name='add_stage_3_4567')([stage_4567_upsample, x_stage3_1x1])  # filters = 64
        stage_34567_upsample = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2),
                                               padding='same', kernel_regularizer=l2(l2_weight),
                                               name='stage_')(stage_34567)
        stage_234567 = Add(name='add_stage_2_34567')([stage_4567_upsample, x_stage2_1x1])

        x_feature_concat= _feature_concat_deconv_branch(C7=x_stage7_1x1, C6=stage_67, C5=stage_567,
                                                            C4=stage_4567, C3=stage_34567, C2=stage_234567)
        x_feature_concat = Conv2D(kernel_size=(1,1), filters=2, kernel_regularizer=l2(l2_weight),
                                  name='all_feature_concat')(x_feature_concat)
        x_output = Activation('softmax', name='Final_Softmax')(x_feature_concat)
        detnet_model = Model(inputs=img_input,
                             outputs=x_output)
        return detnet_model

    ####################################
    # FCN 27 as in paper sfcn-opi
    ####################################
    def fcn27_backbone(self, l2_weight):
        #######################################
        # Extra branch for every pyramid feature
        #######################################
        def feature_prediction_deconv_branch(self, C6=None, C5=None, C4=None, C3=None, C2=None):
            """

            :param features: input feature is from every feature pyramid layer,
                             should've been already connect with 1x1 convolution layer.
            """

            def _64to512(input, type):
                x_deconv128 = Conv2DTranspose(kernel_size=(3, 3),
                                              filters=256, strides=(2, 2), name=type + '_deconv_1')(input)
                x_deconv128 = BatchNormalization()(x_deconv128)
                x_deconv128 = Activation('relu')(x_deconv128)
                x_deconv256 = Conv2DTranspose(kernel_size=(3, 3),
                                              filters=256, strides=(2, 2), name=type + '_deconv_2')(x_deconv128)
                x_deconv256 = BatchNormalization()(x_deconv256)
                x_deconv256 = Activation('relu')(x_deconv256)
                x_deconv512 = Conv2DTranspose(kernel_size=(3, 3),
                                              filters=2, strides=(2, 2), name=type + '_deconv_3')(x_deconv256)
                return x_deconv512

            # in detnet setting, C6.shape == C5.shape == C4.shape, in fcn27 this is 1/4 of origin image
            C6_deconv = _64to512(C6, 'C6')
            C5_deconv = _64to512(C5, 'C5')
            C4_deconv = _64to512(C4, 'C4')
            C3_deconv_256 = Conv2DTranspose(kernel_size=(3, 3),
                                            filters=256, strides=(2, 2), name='C3_deconv_1')(C3)
            C3_deconv_256 = BatchNormalization()(C3_deconv_256)
            C3_deconv_256 = Activation('relu')(C3_deconv_256)
            C3_deconv_512 = Conv2DTranspose(kernel_size=(3, 3), filters=2, strides=(2, 2), name='C3_deconv_2')(
                C3_deconv_256)

            C2_deconv_512 = Conv2DTranspose(kernel_size=(3, 3),
                                            filters=2, strides=(2, 2), name='C2_deconv_1')(C2)

            C23456_concat = Concatenate()([C6_deconv, C5_deconv, C4_deconv, C3_deconv_512, C2_deconv_512])

            return C23456_concat
        #tf.reset_default_graph()
        img_input = Input(self.input_shape)
        #########
        # Adapted first stage
        #########
        x_stage1 = self.first_layer(inputs=img_input)
        x_stage2, x_stage3, x_stage4 = fcn_27(x_stage1, l2_weight= l2_weight, stages=[2, 3, 4])
        # stage3 is 1/2 size
        #########
        # following layer proposed by DetNet
        #########
        x_stage5_B = dilated_with_projection(x_stage4, stage=5, l2_weight= l2_weight)
        x_stage5_A1 = dilated_bottleneck(x_stage5_B, stage=5, block=1, l2_weight= l2_weight)
        x_stage5 = dilated_bottleneck(x_stage5_A1, stage=5, block=2, l2_weight= l2_weight)
        x_stage6_B = dilated_with_projection(x_stage5, stage=6, l2_weight= l2_weight)
        x_stage6_A1 = dilated_bottleneck(x_stage6_B, stage=6, block=1, l2_weight= l2_weight)
        x_stage6 = dilated_bottleneck(x_stage6_A1, stage=6, block=2, l2_weight= l2_weight)
        x_stage2_1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                              name='stage2_1x1_conv')(x_stage2)
        x_stage3_1x1 = Conv2D(filters=64, kernel_size=(1,1), padding='same',
                              name = 'stage3_1x1_conv',
                              kernel_regularizer=l2(l2_weight))(x_stage3)
        # stage4 is 1/4 size, same as 5 and 6
        x_stage4_1x1 = Conv2D(filters=128, kernel_size=(1,1), padding='same',
                              name = 'stage4_1x1_conv',
                              kernel_regularizer=l2(l2_weight))(x_stage4)
        x_stage5_1x1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                              name='stage5_1x1_conv', kernel_regularizer=l2(l2_weight))(x_stage5)
        x_stage6_1x1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                              name='stage6_1x1_conv', kernel_regularizer=l2(l2_weight))(x_stage6)
        
        stage_456 = Add(name='add_stage4_5_6')([x_stage6_1x1, x_stage5_1x1, x_stage4_1x1])
        stage_456_upsample = Conv2DTranspose(filters=128, kernel_size=(1, 1), strides=(2, 2),
                                             name='stage456_upsample')(stage_456)
        stage_3456 = Add(name='stage3_add_456')([stage_456_upsample, x_stage3_1x1])
        stage_3456_upsample = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2),
                                              padding='same',
                                              kernel_regularizer=keras.regularizers.l2(l2_weight),
                                              name='stage3456_upsample')(stage_3456)
        stage_23456 = Add(name='stage2_add_3456')([stage_3456_upsample, x_stage2_1x1]) # filters = 64
        stage_23456_upsample = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2),
                                              padding='same', kernel_regularizer=l2(l2_weight),
                                              name='Deconv_b4_softmax_output')(stage_23456)


        
        x_output_b4_softmax = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2),
                                              padding='same', kernel_regularizer=l2(l2_weight),
                                              name='Deconv_b4_softmax_output')(stage_34)
        
        x_output = Activation('softmax', name='Final_Softmax')(x_output_b4_softmax)
        detnet_model = Model(inputs=img_input,
                             outputs=x_output)
        return detnet_model

    