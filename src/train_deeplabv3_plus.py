from util import load_data, set_gpu, set_num_step_and_aug, lr_scheduler, aug_on_fly, heavy_aug_on_fly
from keras.optimizers import SGD
from encoder_decoder_object_det import data_prepare, save_model_weights, tune_loss_weight, TimerCallback
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from config import Config
from keras.utils import np_utils
from deeplabv3_plus import load_pretrain_weights, preprocess_input, Deeplabv3
from loss import classification_loss
import os
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia

weight_decay = 0.005
epsilon = 1e-7

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'cls_and_det')
CROP_DATA_DIR = os.path.join(ROOT_DIR, 'crop_cls_and_det')
TENSORBOARD_DIR = os.path.join(ROOT_DIR, 'tensorboard_logs')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoint')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'model_weights')


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
    earlystop_callback = EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       min_delta=0.001)
    return [tensorboard_callback, checkpoint_callback, timer, earlystop_callback]


def crop_shape_generator_with_heavy_aug(features, cls_labels, batch_size,
                                       aug_num=25):
    batch_features = np.zeros((batch_size * aug_num, 256, 256, 3))
    batch_cls_labels = np.zeros((batch_size * aug_num, 256, 256, 5))
    while True:
        counter = 0
        for i in range(batch_size):
            index = np.random.choice(features.shape[0], 1)
            feature_index = features[index]
            cls_label_index = cls_labels[index]

            for k in range(aug_num):
                aug_feature, aug_cls_label= heavy_aug_on_fly(feature_index, cls_label_index)
                batch_features[counter] = aug_feature
                batch_cls_labels[counter] = aug_cls_label
                counter = counter + 1

        yield batch_features, batch_cls_labels


def heavy_aug_on_fly(img, mask):
    """Do augmentation with different combination on each training batch
    """

    def image_heavy_augmentation(image, masks, ratio_operations=0.6):
        # according to the paper, operations such as shearing, fliping horizontal/vertical,
        # rotating, zooming and channel shifting will be apply
        hor_flip_angle = np.random.uniform(0, 1)
        ver_flip_angle = np.random.uniform(0, 1)
        seq = iaa.Sequential([
            iaa.SomeOf((0, 5), [
                iaa.Fliplr(hor_flip_angle),
                iaa.Flipud(ver_flip_angle),
                iaa.Affine(shear=(-16, 16)),
                iaa.Affine(scale={'x': (1, 1.6), 'y': (1, 1.6)}),
                iaa.PerspectiveTransform(scale=(0.01, 0.1)),
            ])])

        seq_to_deterministic = seq.to_deterministic()
        aug_img = seq_to_deterministic.augment_images(image)
        aug_mask = seq_to_deterministic.augment_images(masks)
        return aug_img, aug_mask

    aug_image, aug_cls_mask = image_heavy_augmentation(img, mask)
    return aug_image, aug_cls_mask


def data_prepare(print_image_shape=False, print_input_shape=False):
    """
    prepare data for model.
    :param print_image_shape: print image shape if set true.
    :param print_input_shape: print input shape(after categorize) if set true
    :return: list of input to model
    """
    def reshape_mask(origin, cate, num_class):
        return cate.reshape((origin.shape[0], origin.shape[1], origin.shape[2], num_class))

    train_imgs, train_det_masks, train_cls_masks = load_data(data_path=CROP_DATA_DIR, type='train', det=False, cls=True)
    valid_imgs, valid_det_masks, valid_cls_masks = load_data(data_path=CROP_DATA_DIR, type='validation', det=False, cls=True)
    test_imgs, test_det_masks, test_cls_masks = load_data(data_path=CROP_DATA_DIR, type='test',det=False, cls=True)

    if print_image_shape:
        print('Image shape print below: ')
        print('train_imgs: {}, train_cls_masks: {}'.format(train_imgs.shape, train_cls_masks.shape))
        print('valid_imgs: {}, valid_cls_masks: {}'.format(valid_imgs.shape, valid_cls_masks.shape))
        print('test_imgs: {}, test_cls_masks: {}'.format(test_imgs.shape, test_cls_masks.shape))
        print()

    train_cls = np_utils.to_categorical(train_cls_masks, 5)
    train_cls = reshape_mask(train_cls_masks, train_cls, 5)

    valid_cls = np_utils.to_categorical(valid_cls_masks, 5)
    valid_cls = reshape_mask(valid_cls_masks, valid_cls, 5)

    test_cls = np_utils.to_categorical(test_cls_masks, 5)
    test_cls = reshape_mask(test_cls_masks, test_cls, 5)

    if print_input_shape:
        print('input shape print below: ')
        print('train_imgs: {}, train_det: {}'.format(train_imgs.shape, train_cls.shape))
        print('valid_imgs: {}, valid_det: {}'.format(valid_imgs.shape, valid_cls.shape))
        print('test_imgs: {}, test_det: {}'.format(test_imgs.shape, test_cls.shape))
        print()
    return [train_imgs, train_cls, valid_imgs, valid_cls,  test_imgs, test_cls]

if __name__ == '__main__':
    set_gpu()
    hyper_para = tune_loss_weight()
    BATCH_SIZE = Config.image_per_gpu * Config.gpu_count
    print('batch size is :', BATCH_SIZE)
    EPOCHS = Config.epoch

    NUM_TO_AUG, TRAIN_STEP_PER_EPOCH = set_num_step_and_aug()

    data = data_prepare(print_input_shape=True, print_image_shape=True)
    network = Deeplabv3()
    optimizer = SGD(lr=0.01, decay=0.00001, momentum=0.9, nesterov=True)
    hyper = '{}_loss:{}_lr:0.01'.format('Deeplabv3+', Config.model_loss)
    model_weights_saver = save_model_weights(hyper)
    if not os.path.exists(model_weights_saver):
        #loss_input =
        network.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['accuracy'])
        network.summary()

        list_callback = callback_preparation(network, hyper)
        list_callback.append(LearningRateScheduler(lr_scheduler))
        network.fit_generator(crop_shape_generator_with_heavy_aug(data[0],
                                                                 data[1],
                                                                 batch_size=BATCH_SIZE,
                                                                 aug_num=NUM_TO_AUG),
                              epochs=EPOCHS,
                              steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                              validation_data=crop_shape_generator_with_heavy_aug(data[2],
                                                                                data[3],
                                                                                batch_size=BATCH_SIZE,
                                                                                aug_num=NUM_TO_AUG),
                              validation_steps=10,
                              callbacks=list_callback)

        network.save_weights(model_weights_saver)
