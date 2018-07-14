import numpy as np
import tensorflow as tf
import keras as K
import warnings
from keras.layers import BatchNormalization, Input, Conv2D, Activation, Add, Conv2DTranspose, Merge

from keras.optimizers import SGD
from keras.initializers import glorot_uniform
from keras.models import Model, Sequential
from keras.regularizers import l2

class SCFNnetwork:
    def __init__(self, input_shape = (64, 64, 3)):
        #self.inputs = inputs
        self.input_shape = input_shape

    def first_layer(self, inputs, kernel_size):
        """
        First convolution layer.
        """
        X = Conv2D(filters=32, padding='same', kernel_size=(kernel_size, kernel_size), input_shape = self.input_shape,
                   name='conv_first_layer')(inputs)
        X = BatchNormalization(name='bn_first_layer')(X)
        X = Activation('relu')(X)
        return X

    def identity_block(self, f, kernel_size, stage, block, inputs):
        """
        :param f: number of filters
        :param stage: stage of residual blocks
        :param block: ith module
        """
        X_shortcut = inputs

        X = Conv2D(filters=f, padding='same', kernel_size = (kernel_size, kernel_size),
                   name='conv_1a_' + str(stage)+ '_' + str(block))(inputs)
        X = BatchNormalization(name='bn_1b_' + str(stage) + '_' + str(block))(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=f, padding='same', kernel_size=(kernel_size, kernel_size),
                   name='conv_2a_idblcok_' + str(stage) + '_' + str(block))(X)
        X = BatchNormalization(name='bn_2b_' + str(stage) + '_' + str(block))(X)
        X = Add(name='res_identity' + str(stage) +'_' + str(block))([X, X_shortcut])
        X = Activation('relu')(X)
        return X

    def convolution_block(self, f, kernel_size, stage, block, inputs):
        """
        :param f: number of filters
        :param stage: stage of residual blocks
        :param block: ith module
        """
        X = Conv2D(filters=f, kernel_size=(kernel_size, kernel_size),strides=(2,2), padding='same',
                   name='conv_block_1a' + str(stage) + '_' + str(block))(inputs)
        X = BatchNormalization(name='bn_1b_' + str(stage) + '_' + str(block))(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=f, padding='same', kernel_size=(kernel_size, kernel_size),
                   name='conv_2a_' + str(stage) + '_' + str(block))(X)
        X = BatchNormalization(name='bn_2b_' + str(stage) + '_' + str(block))(X)

        X_shortcut = Conv2D(f, kernel_size=(1,1), strides=(2,2), padding='same', name='conv_shortcut_' + str(stage) + '_' + str(block))(inputs)
        X_shortcut = BatchNormalization(name = 'bn_shortcut' + str(block))(X_shortcut)
        print('X: {}, X_shortcut: {}'.format(X.shape, X_shortcut.shape))
        X = Add(name='res_identity' + str(stage) + '_' + str(block))([X, X_shortcut])
        X = Activation('relu')(X)
        return X

    def res_block(self, inputs, filter, kernel_size, stages, block, if_conv = False):
        x = inputs
        if not if_conv:
            for stage in range(stages):
                x = self.identity_block(f=filter, kernel_size=kernel_size,
                                        stage=stage, block=block, inputs=x)
        else:
            for stage in range(stages):
                if stage == 0:
                    x = self.convolution_block(f=filter, kernel_size=kernel_size,
                                               stage=stage, block=block, inputs=inputs)
                else:
                    x = self.identity_block(f=filter, kernel_size = kernel_size,
                                            stage=stage, block=block, inputs=x)
        return x

    def first_and_second_res_blocks(self, inputs, kernel_size):
        """
        Shared residual blocks for detection and classification layers.
        """
        x = self.res_block(inputs, filter=32, kernel_size=kernel_size, stages=9, block=1)
        x = self.res_block(x, filter=64, kernel_size=kernel_size, stages=9, block=2, if_conv=True)
        return x

    def third_res_blocks(self, inputs, kernel_size):
        X = inputs
        for stage in range(1, 10):
            if stage == 1:
                X = self.convolution_block(f=128, kernel_size=kernel_size, stage=stage, block=3, inputs=X)
            else:
                X = self.identity_block(f=128, kernel_size=kernel_size, stage=stage, block=3, inputs=X)
        return X

    def detection_branch(self, kernel_size = 3):
        input_img = Input(shape=self.input_shape)
        X = self.first_layer(input_img, kernel_size)
        X = self.first_and_second_res_blocks(X, kernel_size)

        X_divergent_one = X
        X_divergent_one = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                               name='Conv2D_diverge_one')(X_divergent_one)
        X_divergent_one = BatchNormalization(name='bn_diverge_one')(X_divergent_one)
        X_divergent_one = Activation('relu')(X_divergent_one)

        print(X.shape)
        X_for_future_classification = self.third_res_blocks(X, kernel_size)

        print('X_div_one: ', X_divergent_one.shape, X_for_future_classification.shape)
        #these are detection branch
        X_divergent_two = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                               name='Conv2D_diverge_two')(X_for_future_classification)
        X_divergent_two = BatchNormalization(name='bn_conv_diverge_two')(X_divergent_two)
        X_divergent_two = Activation('relu')(X_divergent_two)
        X_divergent_two = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                        name='Deconv_before_summation')(X_divergent_two)
        X_divergent_two = BatchNormalization()(X_divergent_two)
        X_divergent_two = Activation('relu')(X_divergent_two)


        x_merge = Add()([X_divergent_one, X_divergent_two])
        x_detection = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                  name='Deconv_detection_final_layer')(x_merge)
        x_detection = BatchNormalization()(x_detection)
        x_detection = Activation('softmax', name = 'detection_branch_final_layer')(x_detection)
        print('X_detection', x_detection.shape)
        model = Model(inputs=input_img, outputs=x_detection)
        return model, X_for_future_classification

    #def fourth_res_block(self):



if __name__ == '__main__':
    print('model summary starts')
    a = SCFNnetwork()
    m, x = a.detection_branch()
    m.summary()
    print(x)
