from encoder_decoder_object_det import Detnet
import scipy.misc as misc
import numpy as np
import matplotlib.pyplot as plt
import os
from config import Config
from metric import non_max_suppression, get_metrics
import scipy.io as sio
import cv2

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

WEIGHT_DIR = os.path.join(ROOT_DIR, 'model_weights')
IMG_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'cls_and_det', 'test')


def eval_single_img(model, img_dir, print_img=True,
                    prob_threshold=None, print_single_result=True):
    image_path = os.path.join(IMG_DIR, img_dir, img_dir+ '.bmp')
    #print(image_path)
    img = misc.imread(image_path)
    img = misc.imresize(img, (512, 512), interp='nearest')
    img = img / 255.
    img -= np.mean(img, keepdims=True)
    img /= (np.std(img, keepdims=True) + 1e-7)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    output = model.predict(img)[0]
    output = output[:, :, 1]
    if print_img:
        plt.imshow(output)
        plt.colorbar()
        plt.show()
    p, r, f1, tp = score_single_img(output, img_dir=img_dir, prob_threshold=prob_threshold,
                                    print_single_result=print_single_result)
    return p, r, f1, tp


def score_single_img(input, img_dir, prob_threshold=None, print_single_result=True):
    input = misc.imresize(input, (500, 500))
    input = input / 255.
    boxes = non_max_suppression(input, prob_thresh=prob_threshold)

    num_of_nuclei = boxes.shape[0]

    mat_path = os.path.join(IMG_DIR, img_dir, img_dir + '_detection.mat')
    gt = sio.loadmat(mat_path)['detection']
    outputbase = cv2.imread(os.path.join(IMG_DIR, img_dir, img_dir + '.bmp'))
    if print_single_result:
        print('----------------------------------')
        print('This is {}'.format(img_dir))
        print('detected: {}, ground truth: {}'.format(num_of_nuclei, gt.shape[0]))
    pred = []
    for i in range(boxes.shape[0]):
        x1 = boxes[i, 0]
        y1 = boxes[i, 1]
        x2 = boxes[i, 2]
        y2 = boxes[i, 3]
        cx = int(x1 + (x2 - x1) / 2)
        cy = int(y1 + (y2 - y1) / 2)
        # cv2.rectangle(outputbase,(x1, y1), (x2, y2),(255,0,0), 1)
        cv2.circle(outputbase, (cx, cy), 3, (255, 255, 0), -1)
        pred.append([cx, cy])
    p, r, f1, tp = get_metrics(gt, pred, print_single_result=print_single_result)
    return p, r, f1, tp


def eval_testset(model, prob_threshold=None, print_img=False, print_single_result=True):
    total_p, total_r, total_f1, total_tp = 0, 0, 0, 0
    for img_dir in os.listdir(IMG_DIR):
        p, r, f1, tp = eval_single_img(model, img_dir, print_img=print_img,
                                       print_single_result=print_single_result,
                                       prob_threshold=prob_threshold)
        total_p += p
        total_r += r
        total_f1 += f1
        total_tp += tp
    if prob_threshold is not None:
        print('The nms threshold is {}'.format(prob_threshold))
    print('Over test set, the average P: {}, R: {}, F1: {}, TP: {}'.format(total_p/20,
                                                                           total_r/20,
                                                                           total_f1/20,
                                                                           total_tp/20))
    return total_p/20, total_r/20, total_f1/20

def eval_weights_testset(weightsdir):
    weights_dict = {'best_p': 0, 'best_r': 0, 'best_f1': 0,
                    'best_p_model': None, 'best_r_mode': None,
                    'best_f1_model':None, 'best_p_prob': None,
                    'best_r_prob': None, 'best_f1_prob': None}
    prob_threshhold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print('Start model evalutation')
    for i, weight_dir in enumerate(os.listdir(weightsdir)):
        if 'resnet50' in weight_dir:
            model = Detnet().detnet_resnet50_backbone()
        elif 'resnet101' in weight_dir:
            model = Detnet().detnet_resnet101_backbone()
        elif 'resnet152' in weight_dir:
            model = Detnet().detnet_resnet152_backbone()

        model.load_weights(os.path.join(WEIGHT_DIR, weight_dir))
        for prob in prob_threshhold:
            print('###################')
            print('current model is {} at threshold {}'.format(weight_dir, prob))
            avg_p, avg_r, avg_f1 = eval_testset(model, prob_threshold=prob, print_img=False,
                                                print_single_result=False)
            if weights_dict['best_p'] == 0:
                weights_dict['best_p'], weights_dict['best_r'], weights_dict['best_f1'] = avg_p, avg_r, avg_f1
            else:
                if avg_p > weights_dict['best_p'] and avg_r > 0.6 and avg_f1 > 0.6:
                    weights_dict['best_p'] = avg_p
                    weights_dict['best_p_model'] = weight_dir
                    weights_dict['best_p_prob'] = prob
                if avg_r > weights_dict['best_r'] and avg_p > 0.6 and avg_f1 > 0.6:
                    weights_dict['best_r'] = avg_r
                    weights_dict['best_r_model'] = weight_dir
                    weights_dict['best_r_prob'] = prob
                if avg_f1 > weights_dict['best_f1'] and avg_r > 0.6 and avg_p > 0.6:
                    weights_dict['best_f1'] = avg_f1
                    weights_dict['best_f1_model'] = weight_dir
                    weights_dict['best_f1_prob'] = prob

        print('best precision is {} with model {} at threshold {}'.format(weights_dict['best_p'],
                                                                          weights_dict['best_p_model'],
                                                                          weights_dict['best_p_prob']))
        print('best recall is {} with model {} at threshold {}'.format(weights_dict['best_r'],
                                                                       weights_dict['best_p_model'],
                                                                       weights_dict['best_p_prob']))
        print('best f1 is {} with model {} at threshold {}'.format(weights_dict['best_f1'],
                                                                   weights_dict['best_f1_model'],
                                                                   weights_dict['best_f1_prob']))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.gpu)
    eval_weights_testset(WEIGHT_DIR)
    """
    weight_path = 'base_resnet50_loss:base_lr:0.01_bkg:0.2__train.h5'
    imgdir = 'img' + str(29)
    model = Detnet().detnet_resnet50_backbone()
    model.load_weights(os.path.join(WEIGHT_DIR, weight_path))
    prob_threshhold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    eval_single_img(model, imgdir)
    """
    """
    for prob in prob_threshhold:
        print('The nms threshold is ', prob)
        eval_testset(model, prob_threshold=prob)
"""









'''
    print(p, r, f1)
    tp_num += tp
    gt_num += gt.shape[0]
    pred_num += np.array(pred).shape[0]
precision = tp_num / (pred_num + epsilon)
recall = tp_num / (gt_num + epsilon)
f1_score = 2 * (precision * recall / (precision + recall + epsilon))

print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1_score)
'''

#model = Detnet(0.25).detnet_backbone()
#model.load_weights('../model_weights/focal_aug_elas_trans_smooth_4_bkg_1_train.h5')
#img = misc.imread('img90.bmp')

