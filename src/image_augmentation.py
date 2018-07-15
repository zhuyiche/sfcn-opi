import os
from data_manager import DataManager as dm
import cv2
from numpy.random import randint
import numpy as np
from imgaug import augmenters as iaa


class ImageCropping:
    def __init__(self, data_path = None, old_filename = None, new_filename = None):
        self.data_path = data_path
        self.old_filename = '{}/{}'.format(data_path, old_filename)
        self.new_filename = '{}/{}'.format(data_path, new_filename)
        dm.check_directory(self.new_filename)
        dm.initialize_train_test_folder(self.new_filename)

    @staticmethod
    def crop_image_batch(image, masks=None, if_mask=True, if_det = True, if_cls = True,
                         origin_shape=(500, 500), desired_shape=(64, 64)):
        assert image.ndim == 4
        ori_width, ori_height = origin_shape[0], origin_shape[1]
        des_width, des_height = desired_shape[0], desired_shape[1]

        max_x = ori_width - des_width
        max_y = ori_height - des_height
        ran_x = np.random.randint(0, max_x)
        ran_y = np.random.randint(0, max_y)
        cropped_x = ran_x + des_width
        cropped_y = ran_y + des_height
        cropped_img = image[:, ran_x:cropped_x, ran_y:cropped_y]
        if if_mask and masks is not None:
            if if_det and if_cls:
                det_mask = masks[0]
                cls_mask = masks[1]
                cropped_det_mask = det_mask[:, ran_x:cropped_x, ran_y:cropped_y]
                cropped_cls_mask = cls_mask[:, ran_x:cropped_x, ran_y:cropped_y]
                return cropped_img, cropped_det_mask, cropped_cls_mask
            elif if_det and not if_cls:
                det_mask = masks
                cropped_det_mask = det_mask[:, ran_x:cropped_x, ran_y:cropped_y]
                return cropped_img, cropped_det_mask
            elif if_cls and not if_det:
                cls_mask = masks
                cropped_cls_mask = cls_mask[:, ran_x:cropped_x, ran_y:cropped_y, :]
                return cropped_img, cropped_cls_mask
        else:
            return cropped_img

    @staticmethod
    def crop_image(image, masks=None, if_mask=True, if_det = True, if_cls = True,
                   origin_shape=(500, 500), desired_shape=(64, 64)):
        assert image.ndim == 3
        ori_width, ori_height = origin_shape[0], origin_shape[1]
        des_width, des_height = desired_shape[0], desired_shape[1]

        max_x = ori_width - des_width
        max_y = ori_height - des_height
        ran_x = randint(0, max_x)
        ran_y = randint(0, max_y)
        cropped_x = ran_x + des_width
        cropped_y = ran_y + des_height
        cropped_img = image[ran_x:cropped_x, ran_y:cropped_y]
        if if_mask and masks is not None:
            if if_det and if_cls:
                det_mask = masks[0]
                cls_mask = masks[1]
                cropped_det_mask = det_mask[ran_x:cropped_x, ran_y:cropped_y]
                cropped_cls_mask = cls_mask[ran_x:cropped_x, ran_y:cropped_y]
                return cropped_img, cropped_det_mask, cropped_cls_mask
            elif if_det and not if_cls:
                det_mask = masks[0]
                cropped_det_mask = det_mask[ran_x:cropped_x, ran_y:cropped_y]
                return cropped_img, cropped_det_mask
            elif if_cls and not if_det:
                cls_mask = masks[0]
                cropped_cls_mask = cls_mask[ran_x:cropped_x, ran_y:cropped_y]
                return cropped_img, cropped_cls_mask
        else:
            return cropped_img

    def image_cropping_process(self, num_crop_img=30, files = ['train', 'test', 'validation']):
        """
        Crop image and masks into 64 * 64 * 3
        :param data_path:
        :param num_crop_img:
        :param files:
        :return:
        """
        for file in files:
            counter = 0
            new_path = '{}/{}'.format(self.new_filename, file)
            old_path = '{}/{}'.format(self.old_filename, file)
            for i, _img_folder in enumerate(os.listdir(old_path)):
                masks = []
                for k, _img_file in enumerate(os.listdir(os.path.join(old_path, _img_folder))):
                    if '_detection.bmp' in _img_file:
                        read_det_mask_path = os.path.join(old_path, _img_folder, _img_file)
                        det_img = cv2.imread(read_det_mask_path)
                        masks.append(det_img)
                    elif '_classification.bmp' in _img_file:
                        read_cls_mask_path = os.path.join(old_path, _img_folder, _img_file)
                        cls_img = cv2.imread(read_cls_mask_path)
                        masks.append(cls_img)
                    elif '_original.bmp' in _img_file:
                        read_img_path = os.path.join(old_path, _img_folder, _img_file)
                        img = cv2.imread(read_img_path)
                #print(read_img_path, read_det_mask_path, read_cls_mask_path)
                for j in range(1, num_crop_img+1):
                    counter = counter + 1
                    dm.check_directory('{}/img{}'.format(new_path, counter))
                    cropped_img, cropped_det_mask, cropped_cls_mask = ImageCropping.crop_image(img, masks=masks)
                    dm.check_cv2_imwrite(os.path.join(new_path, 'img{}'.format(counter), 'img{}_original.bmp'.format(counter)), cropped_img)
                    dm.check_cv2_imwrite(os.path.join(new_path, 'img{}'.format(counter), 'img{}_detection.bmp'.format(counter)), cropped_det_mask)
                    dm.check_cv2_imwrite(os.path.join(new_path, 'img{}'.format(counter), 'img{}_classification.bmp'.format(counter)), cropped_cls_mask)
            print('there are {} cropped images for {}'.format(counter, file))


class ImageAugmentation:
    @staticmethod
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

        seq_to_deterministic = seq.to_deterministic()
        aug_img = seq_to_deterministic.augment_image(image)
        aug_det_mask = seq_to_deterministic.augment_image(det_mask)
        aug_cls_mask = seq_to_deterministic.augment_image(cls_mask)
        return aug_img, aug_det_mask, aug_cls_mask

if __name__ == '__main__':
    path = '/home/yichen/Desktop/sfcn-opi-yichen/CRCHistoPhenotypes_2016_04_28'
    new_file = 'Cropping'
    old_file = 'cls_and_det'
    a = ImageCropping(path, old_file, new_file)
    a.image_cropping_process()
    #b = ImageAugmentation(path, new_file, 'Data_Augmentation')
    #total_img = b.batch_augmentation()
    #print('There are {} samples in this data set'.format(total_img))
    #print(counter)
