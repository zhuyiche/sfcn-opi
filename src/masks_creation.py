from scipy.io import loadmat
import cv2
import numpy as np
from sys import platform
import os


class MaskCreation:
    def __init__(self, data_path):
        """decide which platform for default test.
        """
        self.data_path = data_path

    def __add__(self, other):
        """ add two strings together
        """
        return self + other

    def dataset_path(self, num):
        path1 = self.data_path
        det_path = path1 + 'Detection'
        cls_path = path1 + 'Classification'
        det_img_directory_path = det_path + '/img' + str(num)
        cls_img_directory_path = cls_path + '/img' + str(num)

        # path for the image
        self.img_path = det_img_directory_path + '/img' + str(num) + '.bmp'

        self.det_mat_path = det_path + '/img' + str(num) + '/img' + str(num) + '_detection.mat'
        self.epi_cls_mat_path = cls_path + '/img' + str(num) + '/img' + str(num) + '_epithelial.mat'
        self.fib_cls_mat_path = cls_path + '/img' + str(num) + '/img' + str(num) + '_fibroblast.mat'
        self.inflam_cls_mat_path = cls_path + '/img' + str(num) + '/img' + str(num) + '_inflammatory.mat'
        self.others_cls_mat_path = cls_path + '/img' + str(num) + '/img' + str(num) + '_others.mat'
        #print(self.img_path)
        return [self.img_path, self.det_mat_path, self.epi_cls_mat_path,
                self.fib_cls_mat_path, self.inflam_cls_mat_path, self.others_cls_mat_path]

    def main(self):
        print(self.det_mat_path)

    def exceed_index(self, x, y, width, height, radius):
        if np.round(x) >= width:
            x = width - radius
        if np.round(y) >= height:
            y = height - radius
        return x, y

    def change_pixel_color(self, mask, x, y, width, height, color, radius):
        x, y = self.exceed_index(x, y, width, height, radius)
        cv2.circle(mask, center=(np.round(x).astype(int), np.round(y).astype(int)), radius=radius, color=color,
                   thickness=-1)
        return mask

    def loop_points(self, mask, mat, width, height, color, radius):
        length = mat.shape[0]
        if length == 0:
            return mask
        for i in range(length):
            x = mat[i][0]
            y = mat[i][1]
            new_mask = self.change_pixel_color(mask, x, y, width, height, color=color, radius=radius)
        return new_mask

    def create_mask(self, mats, data_path, order, radius, if_detection=False, shape=(500, 500)):
        height, width = shape[1], shape[0]
        mask = np.zeros(shape, dtype=np.uint8)
        if if_detection:
            mat_det = mats['detection']
        else:
            mat_epi = mats[0]['detection']
            mat_fib = mats[1]['detection']
            mat_inf = mats[2]['detection']
            mat_oth = mats[3]['detection']

        data_path = data_path + 'Cls_and_Det'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        img_path = '{}/img{}'.format(data_path, str(order))
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        mask_path = '{}/mask'.format(img_path)

        if not if_detection:
            cls_mask_path = '{}/classification'.format(mask_path)
            if not os.path.exists(cls_mask_path):
                os.makedirs(cls_mask_path)

            mask = self.loop_points(mask, mat_epi, width, height, (50, 50, 50), radius=radius)
            mask = self.loop_points(mask, mat_fib, width, height, (100, 100, 100), radius=radius)
            mask = self.loop_points(mask, mat_inf, width, height, (150, 150, 150), radius=radius)
            mask = self.loop_points(mask, mat_oth, width, height, (200, 200, 200), radius=radius)

            save_mask = '{}/cls_img{}.png'.format(cls_mask_path, order)
            if not os.path.isfile(save_mask):
                mask = mask * 255
                mask = mask.astype('uint8')
                cv2.imwrite(save_mask, mask)
        else:
            length = mat_det.shape[0]
            det_mask_path = '{}/detection/'.format(mask_path)
            if not os.path.exists(det_mask_path):
                os.makedirs(det_mask_path)
            for i in range(length):
                x = mat_det[i][0]
                y = mat_det[i][1]
                x, y = self.exceed_index(x, y, width, height, radius)
                x, y = np.round(x).astype(int), np.round(y).astype(int)
                cv2.circle(mask, center=(x, y), radius=radius, color=(100, 100, 100), thickness=-1)
            mask = mask.copy()
            save_mask = '{}/det_img{}.png'.format(det_mask_path, order)
            # mask = img_as_ubyte(mask)
            # plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
            # print('type of mask: ', mask.dtype)
            if not os.path.isfile(save_mask):
                mask = mask * 255
                mask = mask.astype('uint8')
                cv2.imwrite(save_mask, mask)
        return mask

    def save_image_to_mask(self, data_path, num_image=101):
        """
        Copy the images in original folder into new folders
        :param num_image:
        :return:
        """
        new_path = '{}/{}'.format(data_path, 'Detection')
        mask_path = '{}/{}'.format(data_path, 'Cls_and_Det')
        for i in range(1, num_image):
            image_path = '{}/img{}/img{}.bmp'.format(new_path, i, i)
            mask_image = '{}/img{}/img{}.bmp'.format(mask_path, i, i)
            img = cv2.imread(image_path)
            if not os.path.exists(mask_image):
                cv2.imwrite(mask_image, img)
    def mat_to_mask(self, sub_folder = ['train', 'test', 'validation'], num=101, radius=2):
        """
        Combine all methods above, convert corrdinates in mat file to masks.
        Each corrdinates are converted to a circle with designated radius.
        :param num:
        :param radius:
        :return:
        """
        for folder in sub_folder:
            images_folder_list = os.listdir(os.path.join(self.data_path, folder))
            for i, img_path in enumerate(images_folder_list):
                mask_folder_path = os.path.join(images_folder_list, 'mask')
                det_mask_folder_path = os.path.join(mask_folder_path, 'detection')
                cls_mask_folder_path = os.path.join(mask_folder_path, 'classification')


        for i in range(1, num):
            mat_list = []
            img_path, det_path, epi_path, fib_path, inflam_path, others_path = self.dataset_path(i)
            print(det_path)
            det_mat = loadmat(det_path)
            print(epi_path)
            epi_mat = loadmat(epi_path)
            fib_mat = loadmat(fib_path)
            inflam_mat = loadmat(inflam_path)
            others_mat = loadmat(others_path)
            det_mask = self.create_mask(det_mat, data_path= self.data_path, order=i, radius=radius, if_detection=True)

            mat_list.append(epi_mat)
            mat_list.append(fib_mat)
            mat_list.append(inflam_mat)
            mat_list.append(others_mat)

            cls_mask = self.create_mask(mat_list, data_path=self.data_path, order=i, radius=radius)

if __name__ == '__main__':
    print('damn')
    path = '/Users/yichen/Desktop/CRCHistoPhenotypes_2016_04_28/'
    a = MaskCreation(path)
    a.mat_to_mask()
    a.save_image_to_mask(a.data_path)