import os
import numpy as np
from keras import models
from model import SCFNnetwork

epsilon = 1e-7
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)


def get_metrics(gt, pred, r=3):
    # calculate precise, recall and f1 score
    gt = np.array(gt).astype('int')
    if pred == []:
        if gt.shape[0] == 0:
            return 1, 1, 1
        else:
            return 0, 0, 0


    pred = np.array(pred).astype('int')

    temp = np.concatenate([gt, pred])

    if temp.shape[0] != 0:
        x_max = np.max(temp[:, 0]) + 1
        y_max = np.max(temp[:, 1]) + 1

        gt_map = np.zeros((y_max, x_max), dtype='int')
        for i in range(gt.shape[0]):
            x = gt[i, 0]
            y = gt[i, 1]
            x1 = max(0, x-r)
            y1 = max(0, y-r)
            x2 = min(x_max, x+r)
            y2 = min(y_max, y+r)
            gt_map[y1:y2,x1:x2] = 1

        pred_map = np.zeros((y_max, x_max), dtype='int')
        for i in range(pred.shape[0]):
            x = pred[i, 0]
            y = pred[i, 1]
            pred_map[y, x] = 1

        result_map = gt_map * pred_map
        tp = result_map.sum()

        precision = tp / (pred.shape[0] + epsilon)
        recall = tp / (gt.shape[0] + epsilon)
        f1_score = 2 * (precision * recall / (precision + recall + epsilon))

        return precision, recall, f1_score

model_save_path = '/home/yichen/Desktop/sfcn-opi-yichen/logs/sfcn-opi-detection_train_model.h5'
test_img = '/home/yichen/Desktop/sfcn-opi-yichen/CRCHistoPhenotypes_2016_04_28/cls_and_det/test/img5/img5.bmp'
test_gt = '/home/yichen/Desktop/sfcn-opi-yichen/CRCHistoPhenotypes_2016_04_28/cls_and_det/test/img5/img5_detection.bmp'
det_model = models.load_model(model_save_path, custom_objects={'weighted_loss': SCFNnetwork.detection_loss})

import skimage, cv2
img = cv2.imread(test_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.
img -= np.mean(img,keepdims=True)
img /= (np.std(img,keepdims=True) + 1e-7)

