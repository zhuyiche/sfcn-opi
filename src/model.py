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
from util import load_data
import os, time
from image_augmentation import ImageCropping, ImageAugmentation
from imgaug import augmenters as iaa
import imgaug as ia

weight_decay = 0.005
epsilon = 1e-7
epochs = 20
BATCH_SIZE = 32

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'cls_and_det')
TENSORBOARD_DIR = os.path.join(ROOT_DIR, 'tensorboard_logs')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoint')

class SFCNnetwork:

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
                   name='Conv_1')(inputs)
        x = BatchNormalization(name='BN_1')(x)
        x = Activation('relu', name='act_1')(x)
        return x

    ######################--ResNet components--#####################
    def identity_block(self, f, kernel_size, stage, block, inputs):
        """
        :param f: number of filters
        :param stage: stage of residual blocks
        :param block: ith module
        """
        x_shortcut = inputs

        x = Conv2D(filters=f, padding='same', kernel_size = (kernel_size, kernel_size),
                   name=str(block)+'_'+str(stage) + '_idblock_conv_1')(inputs)
        x = BatchNormalization(name=str(block)+'_'+str(stage) +'_idblock_BN_1')(x)
        x = Activation('relu', name=str(block)+'_'+str(stage) + '_idblock_act_1')(x)

        x = Conv2D(filters=f, padding='same', kernel_size=(kernel_size, kernel_size)
                   ,name=str(block)+'_'+str(stage) + '_idblock_conv_2')(x)
        x = BatchNormalization(name=str(block)+'_'+str(stage) + '_idblock_BN_2')(x)
        x = Add(name=str(block)+'_'+str(stage) + '_idblock_add')([x, x_shortcut])
        x = Activation('relu', name=str(block)+'_'+str(stage)+ '_idblock_act_2')(x)
        return x

    def convolution_block(self, f, kernel_size, stage, block, inputs):
        """
        :param f: number of filters
        :param stage: stage of residual blocks
        :param block: ith module
        """
        x = Conv2D(filters=f, kernel_size=(kernel_size, kernel_size),strides=(2,2), padding='same',
                   name=str(block)+'_'+str(stage) + '_convblock_conv_1')(inputs)
        x = BatchNormalization(name=str(block)+'_'+str(stage) + '_convblock_BN_1')(x)
        x = Activation('relu', name=str(block)+'_'+str(stage) + '_convblock_act_1')(x)

        x = Conv2D(filters=f, padding='same', kernel_size=(kernel_size, kernel_size),
                   name=str(block) + '_' + str(stage) + '_convblock_conv_2')(x)
        x = BatchNormalization(name=str(block)+'_'+str(stage) + '_convblock_BN_2')(x)

        x_shortcut = Conv2D(f, kernel_size=(1,1), strides=(2,2), padding='same',
                            name=str(block)+'_'+str(stage) + '_convblock_shortcut_conv')(inputs)
        x_shortcut = BatchNormalization(name = str(block)+'_'+str(stage) + '_convblock_shortcut_BN_1')(x_shortcut)
        x = Add(name= str(block) + '_'+str(stage) + '_convblock_add')([x, x_shortcut])
        x = Activation('relu', name = str(block) + '_' + str(stage) + '_convblock_merge_act')(x)
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
                    x = self.identity_block(f=filter, kernel_size=kernel_size,
                                            stage=stage, block=block, inputs=x)
        return x
    ##################--FCN-BACKBONE--####################
    def first_and_second_res_blocks(self, inputs, first_filter, second_filter, kernel_size):
        """
        Shared residual blocks for detection and classification layers.
        """
        x = self.res_block(inputs, filter=first_filter, kernel_size=kernel_size, stages=9, block=1)
        x = self.res_block(x, filter=second_filter, kernel_size=kernel_size, stages=9, block=2, if_conv=True)
        return x

    def classification_branch(self, input, filter):
        """classification branch, sepreate from detection branch.
        """
        x = self.res_block(input, filter=filter, kernel_size=3, stages=9, block=4)
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
        x_output = BatchNormalization()(x)
        #x_output = Activation('softmax', name='Classification_output')(x)
        return x_output

    def model_branch(self, kernel_size=3):
        input_img = Input(shape=self.input_shape)
        x = self.first_layer(input_img, kernel_size)
        x = self.first_and_second_res_blocks(x, 32, 64, kernel_size)

        x_divergent_one = x
        x_divergent_one = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                                 name='conv2D_diverge_one')(x_divergent_one)
        x_divergent_one = BatchNormalization(name='bn_diverge_one')(x_divergent_one)
        x_divergent_one = Activation('relu')(x_divergent_one)

        x_for_future_classification = self.res_block(x, filter=128, kernel_size=3, stages=9, block=3, if_conv=True)

        #these are detection branch
        x_divergent_two = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                               name='conv_diverge_two')(x_for_future_classification)
        x_divergent_two = BatchNormalization(name='bn_diverge_two')(x_divergent_two)
        x_divergent_two = Activation('relu')(x_divergent_two)
        x_divergent_two = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                        name='deconv_before_summation')(x_divergent_two)
        x_divergent_two = BatchNormalization(name='bn_deconv_diverge_two')(x_divergent_two)
        x_divergent_two = Activation('relu', name='last_detection_act')(x_divergent_two)

        x_merge = Add(name='merge_two_divergence')([x_divergent_one, x_divergent_two])
        x_detection = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                  name='Deconv_detection_final_layer')(x_merge)
        x_detection = BatchNormalization(name='last_detection_bn')(x_detection)
        # The detection output
        x_detection = Activation('softmax', name = 'Detection_output')(x_detection)
        # The classification output
        x_classification_nobn= self.classification_branch(x_for_future_classification, filter=128)
        x_classification = Activation('softmax', name='Classification_output')(x_classification_nobn)

        model = Model(inputs=input_img,
                      outputs=[x_detection, x_classification])
        return model

    def to_categorical_tensor(self, x3d, n_cls):
        batch_size, n_rows, n_cols = x3d.shape
        x1d = x3d.ravel()
        y1d = np_utils.to_categorical(x1d, num_classes=n_cls)
        y4d = y1d.reshape([batch_size, n_rows, n_cols, n_cls])
        return y4d


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

    train_imgs, train_det_masks, train_cls_masks = load_data(data_path=DATA_DIR, type='train')
    valid_imgs, valid_det_masks, valid_cls_masks = load_data(data_path=DATA_DIR, type='validation')
    test_imgs, test_det_masks, test_cls_masks = load_data(data_path=DATA_DIR, type='test')

    if print_image_shape:
        print('Image shape print below: ')
        print('train_imgs: {}, train_det_masks: {}, train_cls_masks: {}'.format(train_imgs.shape, train_det_masks.shape,
                                                                                train_cls_masks.shape))
        print('valid_imgs: {}, valid_det_masks: {}, validn_cls_masks: {}'.format(valid_imgs.shape, valid_det_masks.shape, valid_cls_masks.shape))
        print('test_imgs: {}, test_det_masks: {}, test_cls_masks: {}'.format(test_imgs.shape, test_det_masks.shape, test_cls_masks.shape))
        print()

    train_det = np_utils.to_categorical(train_det_masks, 2)
    train_det = reshape_mask(train_det_masks, train_det, 2)
    train_cls = np_utils.to_categorical(train_cls_masks, 5)
    train_cls = reshape_mask(train_cls_masks, train_cls, 5)

    valid_det = np_utils.to_categorical(valid_det_masks, 2)
    valid_det = reshape_mask(valid_det_masks, valid_det, 2)
    valid_cls = np_utils.to_categorical(valid_cls_masks, 5)
    valid_cls = reshape_mask(valid_cls_masks, valid_cls, 5)

    test_det = np_utils.to_categorical(test_det_masks, 2)
    test_det = reshape_mask(test_det_masks, test_det, 2)
    test_cls = np_utils.to_categorical(test_cls_masks, 5)
    test_cls = reshape_mask(test_cls_masks, test_cls, 5)

    if print_input_shape:
        print('input shape print below: ')
        print('train_imgs: {}, train_det: {}, train_cls: {}'.format(train_imgs.shape, train_det.shape, train_cls.shape))
        print('valid_imgs: {}, valid_det: {}, validn_cls: {}'.format(valid_imgs.shape, valid_det.shape, valid_cls.shape))
        print('test_imgs: {}, test_det: {}, test_cls: {}'.format(test_imgs.shape, test_det.shape, test_cls.shape))
        print()
    return [train_imgs, train_det, train_cls, valid_imgs, valid_det, valid_cls, test_imgs, test_det, test_cls]


def crop_on_fly(img, det_mask, cls_mask, crop_size):
    """
    Crop image randomly on each training batch
    """
    imgcrop = ImageCropping()
    cropped_img, cropped_det_mask, cropped_cls_mask = imgcrop.crop_image_batch(img, [det_mask, cls_mask], desired_shape=(crop_size, crop_size))
    return cropped_img, cropped_det_mask, cropped_cls_mask


def aug_on_fly(img, det_mask, cls_mask):
    """Do augmentation with different combination on each training batch
    """
    def image_basic_augmentation(image, masks): #ratio_operations=0.5):
        # without additional operations
        # according to the paper, operations such as shearing, fliping horizontal/vertical,
        # rotating, zooming and channel shifting will be apply
        #sometimes = lambda aug: iaa.Sometimes(ratio_operations, aug)

        hor_flip_angle = np.random.uniform(0, 1)
        ver_flip_angle = np.random.uniform(0, 1)
        seq = iaa.Sequential([
            iaa.SomeOf((0, 5), [
                iaa.Fliplr(hor_flip_angle),
                iaa.Flipud(ver_flip_angle),
                iaa.Affine(shear=(-16, 16)),
                iaa.Affine(scale={'x': (1, 1.6), 'y': (1, 1.6)}),
                iaa.PerspectiveTransform(scale=(0.01, 0.1))
            ])
        ])
        det_mask, cls_mask = masks[0], masks[1]
        """
        if image.ndim == 4:
            image = np.array([ia.quokka(size=(image.shape[1], image.shape[2]))], dtype=np.uint8)
            det_mask = np.array([ia.quokka(size=(image.shape[1], image.shape[2]))], dtype=np.uint8)
            cls_mask = np.array([ia.quokka(size=(image.shape[1], image.shape[2]))], dtype=np.uint8)
           """
        seq_to_deterministic = seq.to_deterministic()
        aug_img = seq_to_deterministic.augment_image(image)
        aug_det_mask = seq_to_deterministic.augment_image(det_mask)
        aug_cls_mask = seq_to_deterministic.augment_image(cls_mask)
        return aug_img, aug_det_mask, aug_cls_mask

    aug_image, aug_det_mask, aug_cls_mask = image_basic_augmentation(image=img, masks=[det_mask, cls_mask])
    return aug_image, aug_det_mask, aug_cls_mask


def generator_with_aug(features, det_labels, cls_labels, batch_size, crop_size,
                          crop_num = 30, aug_num = 8):
    batch_features = np.zeros((batch_size * crop_num * aug_num, crop_size, crop_size, 3))
    batch_det_labels = np.zeros((batch_size * crop_num * aug_num, crop_size, crop_size, 2))
    batch_cls_labels = np.zeros((batch_size * crop_num * aug_num, crop_size, crop_size, 5))
    while True:
        counter = 0
        for i in range(batch_size):
            index = np.random.choice(features.shape[0], 1)
            for j in range(crop_num):
                print(features.shape, det_labels.shape, cls_labels.shape)
                print('data type: ', type(features))
                feature_index = features[index]
                det_label_index = det_labels[index]
                cls_label_index = cls_labels[index]
                feature, det_label, cls_label = crop_on_fly(feature_index, det_label_index, cls_label_index, crop_size = crop_size)
                print('feature.shape: ', feature.shape, ' det_label_index: ',det_label.shape)
                for k in range(aug_num):
                    aug_feature, aug_det_label, aug_cls_label = aug_on_fly(feature, det_label, cls_label)
                    batch_features[counter] = aug_feature
                    batch_det_labels[counter] = aug_det_label
                    batch_cls_labels[counter] = aug_cls_label
                    counter = counter + 1
        yield(batch_features, {'Detection_output': batch_det_labels,
                               'Classification_output': batch_cls_labels})


def generator_without_aug(features, det_labels, cls_labels, batch_size, crop_size,
                          crop_num = 30):
    batch_features = np.zeros((batch_size * crop_num, crop_size, crop_size, 3))
    batch_det_labels = np.zeros((batch_size * crop_num, crop_size, crop_size, 2))
    batch_cls_labels = np.zeros((batch_size * crop_num, crop_size, crop_size, 5))
    while True:
        counter = 0
        for i in range(batch_size):
            index = np.random.choice(features.shape[0], 1)
            for j in range(crop_num):
                feature, det_label, cls_label = crop_on_fly(features[index],
                                                            det_labels[index], cls_labels[index], crop_size=crop_size)
                batch_features[counter] = feature
                batch_det_labels[counter] = det_label
                batch_cls_labels[counter] = cls_label
                counter += 1
        yield batch_features, {'Detection_output': batch_det_labels,
                               'Classification_output': batch_cls_labels}


def callback_preparation(model):
    """
    implement necessary callbacks into model.
    :return: list of callback.
    """
    timer = TimerCallback()
    timer.set_model(model)
    tensorboard_callback = TensorBoard(TENSORBOARD_DIR)
    checkpoint_callback = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, 'det_train_point.h5'), save_best_only=True, period=1)
    return [tensorboard_callback, checkpoint_callback, timer]


def model_compile(model_input, summary=False):
    model = model_input.model_branch()
    print('model built')
    if summary:
        model.summary()

    optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(optimizer=optimizer,
                      loss={'Detection_output': SFCNnetwork.detection_loss,
                            'Classification_output': SFCNnetwork.classification_loss},
                      metrics=['accuracy'])
    return model


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = SFCNnetwork()
    data = data_prepare(print_input_shape=True, print_image_shape=True)
    model = model_compile(model)
    timer = TimerCallback()
    timer.set_model(model)
    tensorboard_callback = TensorBoard(TENSORBOARD_DIR)
    checkpoint_callback = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, 'train_point.h5'), save_best_only=True, period=1)
    CROP_SIZE = 64
    BATCH_SIZE = 2
    EPOCHS = 300
    TRAIN_STEP_PER_EPOCH = 30

    model.fit_generator(generator_with_aug(data[0], data[1], data[2], crop_size=CROP_SIZE, batch_size=BATCH_SIZE),
                        epochs=EPOCHS,
                        steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                        validation_data=generator_with_aug(data[3], data[4], data[5], batch_size=2, crop_size=CROP_SIZE),
                        validation_steps=5, callbacks=callback_preparation(model))
    model.save_weights(os.path.join(ROOT_DIR, 'model_weights', 'after300-sfcn-opi-train_model.h5'))
