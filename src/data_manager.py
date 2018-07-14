import os
import cv2


class DataManager:
    """manage train, test, and validation sets from original directory.
    """
    def __init__(self, data_path):
        self.data_path = data_path

    def reorder_img(self, max_num_file = 101):
        """
        rename the image
        :param max_num_file: maximum number of images.
        """
        train_path = self.data_path + '/train'
        test_path = self.data_path + '/test'
        validation_path = self.data_path + '/validation'
        self.check_directory(train_path)
        self.check_directory(test_path)
        self.check_directory(validation_path)

        self.change_img_order('test')
        self.change_img_order('train')
        self.change_img_order('validation')

    def change_img_order(self, file, max_num_file=101):
        """
        helper function, change img number either in train, test,
        or validation folder.
        :param file: train/test/validation
        :param max_num_file: total image
        :return:
        """
        main_file_path = '{}/{}'.format(self.data_path, file)
        counter = 0
        for i in range(1, max_num_file):
            file_path = '{}/img{}'.format(main_file_path, i)
            if os.path.exists(file_path):
                counter += 1
                old_img_path = '{}/img{}.bmp'.format(file_path, i)
                old_det_path = '{}/mask/detection/dec_img{}.png'.format(file_path, i)
                old_cls_path = '{}/mask/classification/cls_img{}.png'.format(file_path, i)

                new_main_file_path = '{}/img{}'.format(main_file_path, counter)
                new_img_path = '{}/img{}.bmp'.format(new_main_file_path, counter)
                new_det_path = '{}/mask/detection/det_img{}.png'.format(new_main_file_path, counter)
                new_cls_path = '{}/mask/classification/cls_img{}.png'.format(new_main_file_path, counter)

                if not old_img_path == new_img_path:
                    os.rename(file_path, new_main_file_path)
                    DataManager.check_directory('{}/{}'.format(new_main_file_path, 'mask'))
                    DataManager.check_directory('{}/{}/{}'.format(new_main_file_path, 'mask', 'detection'))
                    DataManager.check_directory('{}/{}/{}'.format(new_main_file_path, 'mask', 'classification'))
                    #the os.rename has FileNotFound Error
                    try:
                        os.rename(old_det_path, new_det_path)
                        os.rename(old_cls_path, new_cls_path)
                        os.rename(old_img_path, new_img_path)
                    except:
                        pass

    @staticmethod
    def check_directory(file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    @staticmethod
    def check_cv2_imwrite(file_path, file):
        if not os.path.exists(file_path):
            cv2.imwrite(file_path, file)

    @staticmethod
    def initialize_train_test_folder(file_path):
        """
        create new train, test and validation folder if
        it doesn't exist.
        :param file_path: main_path
        """
        train_folder = '{}/{}'.format(file_path, 'train')
        test_folder = '{}/{}'.format(file_path, 'test')
        valid_folder = '{}/{}'.format(file_path, 'validation')
        DataManager.check_directory(train_folder)
        DataManager.check_directory(test_folder)
        DataManager.check_directory(valid_folder)

    @staticmethod
    def get_img_path(data_path, order, img_format = 'bmp'):
        img_path = '{}/img{}/img{}.{}'.format(data_path, order, order, img_format)
        return img_path

    @staticmethod
    def get_mask_path(data_path, order, type, img_format = 'png'):
        if type == 'detection':
            mask_path = '{}/img{}/mask/detection/det_img{}.{}'.format(data_path, order, order, img_format)
        elif type == 'classification':
            mask_path = '{}/img{}/mask/classification/cls_img{}.{}'.format(data_path, order, order, img_format)
        else:
            raise Exception('type is either detection or classification')
        return mask_path

if __name__ == '__main__':
    path = '/Users/yichen/Desktop/CRCHistoPhenotypes_2016_04_28/Cls_and_Det'

    a = DataManager(path)
    a.reorder_img()