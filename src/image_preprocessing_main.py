from masks_creation import MaskCreation
from data_manager import DataManager
from image_augmentation import ImageAugmentation, ImageCropping
import os

def main(path, ori_file_name, crop_file_name,
         aug_file_name, if_augmentation = False,
         augmentation_per_image = 5, ):
    """

    :param path: dataset path
    :param if_augmentation: False if only cropping needed. To reduce train set size
    """
    #masks = MaskCreation(path)
    #masks.mat_to_mask()
    #masks.save_image_to_mask(path)

    #data = DataManager(path)
    #data.reorder_img()

    #img_crop = ImageCropping(path, ori_file_name, crop_file_name)
    #img_crop.image_cropping_process(path)

    #if if_augmentation:
     #   img_aug = ImageAugmentation(path, crop_file_name, aug_file_name)
      #  img_aug.batch_augmentation(augmentation_per_image)

if __name__ == '__main__':
    print('Starting image preprocessing')
    p = '/home/yichen/Desktop/sfcn-opi-yichen/'
    path = '/Users/yichen/Desktop/CRCHistoPhenotypes_2016_04_28'
    ori_file_name = 'Cls_and_Det'
    crop_file_name = 'Cropping'     ##save images to this subfile after cropping
    aug_file_name = 'Data_Augmentation'      ##save images to this subfile after augmentation
    print(os.path.join(path, ori_file_name))
    main(p, ori_file_name, crop_file_name, aug_file_name, if_augmentation = True)