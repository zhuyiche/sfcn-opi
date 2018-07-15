import numpy as np
import tensorflow as tf
import keras as K
import warnings
from keras.callbacks import TensorBoard, EarlyStopping, Callback, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import BatchNormalization, Input, Conv2D, Activation, Add, Conv2DTranspose, Merge
from keras.regularizers import l2
from keras.layers import Input,Conv2D,Add,BatchNormalization,Activation, Lambda
from keras.models import Model
from keras.utils import plot_model,np_utils
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping,TensorBoard,ModelCheckpoint,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from sklearn.metrics import precision_score, recall_score, f1_score
import util
import os, time, keras
from keras.preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa
from image_augmentation import ImageCropping

weight_decay = 0.005
epsilon = 1e-7
epochs = 20
BATCH_SIZE = 32

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'cls_and_det')

class SCFNnetwork:
    def __init__(self, input_shape = (64, 64, 3)):
        #self.inputs = inputs
        self.input_shape = input_shape

    @staticmethod
    def detection_loss(y_true, y_pred):
        true_bkg = y_true[:, :, :, 0:1]
        pred_bkg = K.clip(y_pred[:, :, :, 0:1], epsilon, 1 - epsilon)
        true_obj = y_true[:, :, :, 1:2]
        pred_obj = K.clip(y_pred[:, :, :, 1:2], epsilon, 1 - epsilon)
        lossbg = - true_bkg * K.log(pred_bkg)
        lossobj = -250 * true_obj * K.log(pred_obj)
        loss = K.sum(K.concatenate([lossbg, lossobj]), -1)
        return loss

    @staticmethod
    def classification_loss(y_true, y_pred):
        true_bkg = y_true[:, :, :, 0:1]
        pred_bkg = K.clip(y_pred[:, :, :, 0:1], epsilon, 1 - epsilon)
        true_obj1 = y_true[:, :, :, 1:2]
        pred_obj1 = K.clip(y_pred[:, :, :, 1:2], epsilon, 1 - epsilon)
        true_obj2 = y_true[:, :, :, 2:3]
        pred_obj2 = K.clip(y_pred[:, :, :, 2:3], epsilon, 1 - epsilon)
        true_obj3 = y_true[:, :, :, 3:4]
        pred_obj3 = K.clip(y_pred[:, :, :, 3:4], epsilon, 1 - epsilon)
        true_obj4 = y_true[:, :, :, 4:]
        pred_obj4 = K.clip(y_pred[:, :, :, 4:], epsilon, 1 - epsilon)
        lossbg = - true_bkg * K.log(pred_bkg)
        loss1 = -40 * true_obj1 * K.log(pred_obj1)
        loss2 = -50 * true_obj2 * K.log(pred_obj2)
        loss3 = -40 * true_obj3 * K.log(pred_obj3)
        loss4 = -120 * true_obj4 * K.log(pred_obj4)
        loss = K.sum(K.concatenate([lossbg, loss1, loss2, loss3, loss4]), -1)
        return loss

    def first_layer(self, inputs, kernel_size):
        """
        First convolution layer.
        """
        x = Conv2D(filters=32, padding='same', kernel_size=(kernel_size, kernel_size), input_shape = self.input_shape,
                   name='conv_first_layer')(inputs)
        x = BatchNormalization(name='bn_first_layer')(x)
        x = Activation('relu')(x)
        return x

    def identity_block(self, f, kernel_size, stage, block, inputs):
        """
        :param f: number of filters
        :param stage: stage of residual blocks
        :param block: ith module
        """
        x_shortcut = inputs

        x = Conv2D(filters=f, padding='same', kernel_size = (kernel_size, kernel_size))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=f, padding='same', kernel_size=(kernel_size, kernel_size))(x)
        x = BatchNormalization()(x)
        x = Add()([x, x_shortcut])
        x = Activation('relu')(x)
        return x

    def convolution_block(self, f, kernel_size, stage, block, inputs):
        """
        :param f: number of filters
        :param stage: stage of residual blocks
        :param block: ith module
        """
        x = Conv2D(filters=f, kernel_size=(kernel_size, kernel_size),strides=(2,2), padding='same',
                   name='conv_block_1a' + str(stage) + '_' + str(block))(inputs)
        x = BatchNormalization(name='bn_convblock_1b_' + str(stage) + '_' + str(block))(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=f, padding='same', kernel_size=(kernel_size, kernel_size),
                   name='conv_2a_convblock_' + str(stage) + '_' + str(block))(x)
        x = BatchNormalization(name='bn_convblock_2b_' + str(stage) + '_' + str(block), trainable=True)(x)

        x_shortcut = Conv2D(f, kernel_size=(1,1), strides=(2,2), padding='same', name='conv_shortcut_' + str(stage) + '_' + str(block))(inputs)
        x_shortcut = BatchNormalization(name = 'bn_shortcut' + str(block))(x_shortcut)
        x = Add()([x, x_shortcut])
        x = Activation('relu')(x)
        return x

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

    def first_and_second_res_blocks(self, inputs, first_filter, second_filter, kernel_size):
        """
        Shared residual blocks for detection and classification layers.
        """
        x = self.res_block(inputs, filter=first_filter, kernel_size=kernel_size, stages=9, block=1)
        x = self.res_block(x, filter=second_filter, kernel_size=kernel_size, stages=9, block=2, if_conv=True)
        return x

    def third_res_blocks(self, inputs, filter, kernel_size, block):
        x = inputs
        x = self.res_block(x, filter=filter, kernel_size=kernel_size, stages=9, block=block, if_conv=True)
        return x

    def detection_branch(self, kernel_size = 3):
        input_img = Input(shape=self.input_shape)
        x = self.first_layer(input_img, kernel_size)
        x = self.first_and_second_res_blocks(x, 32, 64, kernel_size)

        x_divergent_one = x
        x_divergent_one = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                                 name='conv2D_diverge_one')(x_divergent_one)
        x_divergent_one = BatchNormalization(name='bn_diverge_one')(x_divergent_one)
        x_divergent_one = Activation('relu')(x_divergent_one)

        x_for_future_classification = self.third_res_blocks(inputs=x, filter=128, kernel_size=3, block=4)

        #these are detection branch
        x_divergent_two = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                               name='conv2D_diverge_two')(x_for_future_classification)
        x_divergent_two = BatchNormalization(name='bn_conv_diverge_two')(x_divergent_two)
        x_divergent_two = Activation('relu')(x_divergent_two)
        x_divergent_two = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                        name='deconv_before_summation')(x_divergent_two)
        x_divergent_two = BatchNormalization(name='bn_deconv_diverge_two')(x_divergent_two)
        x_divergent_two = Activation('relu')(x_divergent_two)

        x_merge = Add()([x_divergent_one, x_divergent_two])
        x_detection = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                  name='Deconv_detection_final_layer')(x_merge)
        x_detection = BatchNormalization()(x_detection)
        x_detection = Activation('softmax', name = 'detection_branch_final_layer')(x_detection)
        print('x_detection shape: {} '.format(x_detection.shape))
        det_model = Model(inputs=input_img, outputs=x_detection)
        return det_model#, input_img, x_for_future_classification

    def fourth_res_block(self, inputs, filter, kernel_size):
        x = self.res_block(inputs, filter=filter, kernel_size=kernel_size, stages=9, block=4)
        return x

    def classification_branch(self, input, ori_inputs):
        x = self.fourth_res_block(input, filter=128, kernel_size=3)
        #all layers before OPI
        x = Conv2D(filters=5, kernel_size=1, padding='same', name='conv2d_after_fourth_res_block')(x)
        x = BatchNormalization(name='bn_after_fourth_res_block')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=5, kernel_size=(3,3),
                            strides=(2, 2), padding='same',
                            name='second_deconv_before_cls')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=5, kernel_size=(3, 3),
                            strides=(2, 2), padding='same',
                            name='first_deconv_before_cls')(x)
        x = BatchNormalization()(x)
        x = Activation('softmax', name='classification')(x)

        cls_model = Model(inputs=ori_inputs, outputs=x)
        return cls_model

    def to_categorical_tensor(self, x3d, n_cls):
        batch_size, n_rows, n_cols = x3d.shape
        x1d = x3d.ravel()
        y1d = np_utils.to_categorical(x1d, num_classes=n_cls)
        y4d = y1d.reshape([batch_size, n_rows, n_cols, n_cls])
        return y4d

    def nms_layer(self, input):
        return tf.image.non_max_suppression()


class PrecisionRecallF1Callback(Callback):

    def __init__(self, validation_data):
        super(PrecisionRecallF1Callback, self).__init__()
        self.validation_data = validation_data

    def on_train_begin(self, logs=None):
        self.val_f1_list = []
        self.val_recall_list = []
        self.val_precision_list = []

    def on_epoch_end(self, epoch, logs={}):
        x_val, y_val = self.validation_data[0], self.validation_data[1]
        #print(self.validation_data)
        y_predict = self.model.predict(x_val)
        val_recall= recall_score(y_val, y_predict)
        self.val_recall_list.append(val_recall)
        print('val_recall: {}'.format(val_recall))


class TimerCallback(Callback):
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


def crop_on_fly(img, mask):
    #crop = iaa.Sequential(
    #    iaa.Crop(percent=(img.shape[1]/64, img.shape[1]/64))
    #)
    '''
    crop_deterministic = crop.to_deterministic()
    cropped_img = crop_deterministic.augment_image(img)
    cropped_mask = crop_deterministic.augment_image(mask)
    '''
    imgcrop = ImageCropping()
    cropped_img, cropped_mask = imgcrop.crop_image_batch(img, mask, if_cls=False)
    return cropped_img, cropped_mask


def generator_without_augmentation(features, labels, batch_size, cropping_num = 30):
    batch_features = np.zeros((batch_size * cropping_num, 64, 64, 3))
    batch_labels = np.zeros((batch_size * cropping_num, 64, 64, 2))
    #print('features.shape: {} and {}'.format(features.shape[0], features.shape))
    while True:
        counter = 0
        for i in range(batch_size):
            index = np.random.choice(features.shape[0], 1)
            for j in range(cropping_num):
                feature, label = crop_on_fly(features[index], labels[index])
                batch_features[counter] = feature
                batch_labels[counter] = label
                counter += 1
        yield batch_features, batch_labels


def path_setting():
    root_dir = os.getcwd()
    print(root_dir)

def compile(model):
    optimizer = SGD(lr = 0.01, momentum=0.9, decay=0.001, nesterov=True, )
    model.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
                      loss=SCFNnetwork.detection_loss, metrics=['accuracy'])

if __name__ == '__main__':
    tensorboard = TensorBoard('/home/yichen/Desktop/sfcn-opi-yichen/logs/det_train_l2_point')
    lr_reduce = ReduceLROnPlateau(monitor='loss', patience=50, verbose=1)
    checkpoint = ModelCheckpoint('/home/yichen/Desktop/sfcn-opi-yichen/checkpoint/det_trainl2_point.h5', period=1, save_best_only=True)
    earlystopping = EarlyStopping(patience=5)

    print('model summary starts')
    a = SCFNnetwork()
    from util import load_data
    train_imgs, train_det_masks, train_cls_masks = load_data(data_path=DATA_DIR, type='train')
    valid_imgs, valid_det_masks, valid_cls_masks = load_data(data_path=DATA_DIR, type='validation')
    test_imgs, test_det_masks, test_cls_masks = load_data(data_path=DATA_DIR, type='test')
    test_det = np_utils.to_categorical(test_det_masks, 2)
    test_cls = np_utils.to_categorical(test_cls_masks, 5)
    #train_det_masks = train_det_masks((train_det_masks.shape[0], train_det_masks.shape[1], train_det_masks.shape[2], -1))
    #print(train_det_masks.shape)
    train_det = keras.utils.to_categorical(train_det_masks, 2)
    train_cls = np_utils.to_categorical(train_cls_masks, 5)
    valid_det = np_utils.to_categorical(valid_det_masks, 2)
    valid_cls = np_utils.to_categorical(valid_cls_masks, 5)


    PRF = PrecisionRecallF1Callback([valid_imgs, valid_det_masks])
    #print('train_imgs: {}, train_det_masks: {}, train_cls_masks: {}'.format(train_imgs.shape, train_det.shape, train_cls.shape))
    #print('valid_imgs: {}, valid_det: {}, validn_cls: {}'.format(valid_imgs.shape, valid_det.shape, valid_cls.shape))
    #print('test_imgs: {}, test_det: {}, test_cls: {}'.format(test_imgs.shape, test_det.shape, test_cls.shape))
    #det_model, ori_input, x_for_cls = a.detection_branch()

    det_model = a.detection_branch()
    #det_model.summary()

    det_model.compile(optimizer=SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True),
                      loss=SCFNnetwork.detection_loss, metrics=['accuracy'])
    TIMER = TimerCallback()
    TIMER.set_model(det_model)
    PRF.set_model(det_model)

    #print('train_imgs.shape: {}, valid_imgs.shape: {}'.format(train_imgs.shape, valid_imgs.shape))
    #print('train_det_masks.shape-train_cls_masks.shape: {}-{}'.format(train_det_masks.shape, train_cls_masks.shape))
    #print('valid_det.shape-valid_cls.shape:{}-{}'.format(valid_det_masks.shape, valid_cls_masks.shape))
    print('detection branch training starts')
    #a ,b = generator_without_augmentation(train_imgs, train_det_masks, batch_size=BATCH_SIZE)
    start_time = time.time()
    det_model.load_weights('/home/yichen/Desktop/sfcn-opi-yichen/logs/sfcn-opi-detection_train_model.h5')
    det_model.fit_generator(generator_without_augmentation(train_imgs, train_det, batch_size= 2), epochs=150,
                            steps_per_epoch=35,
                            validation_data=generator_without_augmentation(valid_imgs, valid_det, 2),
                            validation_steps=5, callbacks=[TIMER, tensorboard, checkpoint])
    print('detection branch training end, take : {}'.format(int(time.time() - start_time)*1000))
    det_model.save(os.path.join(ROOT_DIR, 'logs', 'after300-sfcn-opi-detection_train_model.h5'))
"""
    det_model.fit(x=train_imgs, y = train_det_masks,
                  epochs=epochs, batch_size=BATCH_SIZE, #callbacks=[PRF],
                  validation_data=[valid_imgs, valid_det])
                  #callbacks=[PRF])
    print('Detection Branch Training Finished...')
    print('Starting Training Classification Branch...')
"""
#print(cls_model.output.sha)