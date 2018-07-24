import os
from data_manager import DataManager as dm
import cv2
from numpy.random import randint
import numpy as np
from imgaug import augmenters as iaa
from util import check_directory,check_cv2_imwrite

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

OLD_DATA_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'cls_and_det')
TRAIN_OLD_DATA_DIR = os.path.join(OLD_DATA_DIR, 'train')
TEST_OLD_DATA_DIR = os.path.join(OLD_DATA_DIR, 'test')
VALID_OLD_DATA_DIR = os.path.join(OLD_DATA_DIR, 'validation')
TARGET_DATA_DIR = os.path.join(ROOT_DIR, 'crop_cls_and_det')
TRAIN_TARGET_DATA_DIR = os.path.join(TARGET_DATA_DIR, 'train')
TEST_TARGET_DATA_DIR = os.path.join(TARGET_DATA_DIR, 'test')
VALID_TARGET_DATA_DIR = os.path.join(TARGET_DATA_DIR, 'validation')


def crop_image_parts(image, det_mask, origin_shape=(512, 512)):
    assert image.ndim == 3
    ori_width, ori_height = origin_shape[0], origin_shape[1]
    des_width, des_height = 256, 256
    assert des_width == des_height

    cropped_img1 = image[0: des_width, 0: des_height, :]  # 1, 3
    cropped_img2 = image[des_width: ori_width, 0: des_height, :]  # 2, 4
    cropped_img3 = image[0: des_width, des_height: ori_height, :]
    cropped_img4 = image[des_width: ori_width, des_height: ori_height, :]

    cropped_mask1 = det_mask[0: des_width, 0: des_height, :]              # 1, 3
    cropped_mask2 = det_mask[des_width: ori_width, 0: des_height, :]      # 2, 4
    cropped_mask3 = det_mask[0: des_width, des_height: ori_height, :]
    cropped_mask4 = det_mask[des_width: ori_width, des_height: ori_height, :]
    return [cropped_img1, cropped_img2, cropped_img3, cropped_img4,
            cropped_mask1, cropped_mask2, cropped_mask3, cropped_mask4]


def batch_crop_image_parts(ori_set, target_set):
    for file in os.listdir(ori_set):
        print(file)
        image_file = os.path.join(ori_set, str(file), str(file) + '.bmp')
        mask_file = os.path.join(ori_set, str(file), str(file) + '_detection.bmp')
        image = cv2.imread(image_file)
        image = cv2.resize(image, (512, 512))
        det_mask = cv2.imread(mask_file)
        det_mask = cv2.resize(det_mask, (512, 512))
        crop_list = crop_image_parts(image, det_mask)

        list_file_create = [os.path.join(target_set, str(file)+'_1'),
                            os.path.join(target_set, str(file)+'_2'),
                            os.path.join(target_set, str(file)+'_3'),
                            os.path.join(target_set, str(file)+'_4')]
        check_directory(list_file_create)
        list_img_create = [os.path.join(target_set, str(file)+ '_1', str(file)+'_1.bmp'),
                           os.path.join(target_set, str(file)+ '_2', str(file)+'_2.bmp'),
                           os.path.join(target_set, str(file)+ '_3', str(file)+'_3.bmp'),
                           os.path.join(target_set, str(file)+'_4', str(file)+'_4.bmp'),
                           os.path.join(target_set, str(file)+'_1',str(file)+'_1_detection.bmp'),
                           os.path.join(target_set, str(file)+'_2',str(file)+'_2_detection.bmp'),
                           os.path.join(target_set, str(file)+'_3',str(file)+'_3_detection.bmp'),
                           os.path.join(target_set, str(file)+'_4',str(file)+'_4_detection.bmp')]
        backup_img_create = [os.path.join(target_set, str(file)+ '_1', str(file)+'_1_original.bmp'),
                           os.path.join(target_set, str(file)+ '_2', str(file)+'_2_original.bmp'),
                           os.path.join(target_set, str(file)+ '_3', str(file)+'_3_original.bmp'),
                           os.path.join(target_set, str(file)+'_4', str(file)+'_4_original.bmp'),
                           os.path.join(target_set, str(file)+'_1',str(file)+'_1_detection.bmp'),
                           os.path.join(target_set, str(file)+'_2',str(file)+'_2_detection.bmp'),
                           os.path.join(target_set, str(file)+'_3',str(file)+'_3_detection.bmp'),
                           os.path.join(target_set, str(file)+'_4',str(file)+'_4_detection.bmp')]
        for order, img in enumerate(crop_list):
            check_cv2_imwrite(list_img_create[order], img)
            check_cv2_imwrite(backup_img_create[order], img)
        #check_directory(list_file_create)
        #cv2.imwrite

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


if __name__ == '__main__':
    batch_crop_image_parts(TRAIN_OLD_DATA_DIR, TRAIN_TARGET_DATA_DIR)
    batch_crop_image_parts(TEST_OLD_DATA_DIR, TEST_TARGET_DATA_DIR)
    batch_crop_image_parts(VALID_OLD_DATA_DIR, VALID_TARGET_DATA_DIR)
