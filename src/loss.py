import keras.backend as K
import tensorflow as tf

epsilon = 1e-7
def detection_loss(y_true, y_pred):
    true_bkg = y_true[:, :, :, 0:1]
    pred_bkg = K.clip(y_pred[:, :, :, 0:1], epsilon, 1 - epsilon)
    true_obj = y_true[:, :, :, 1:2]
    pred_obj = K.clip(y_pred[:, :, :, 1:2], epsilon, 1 - epsilon)
    lossbg = - true_bkg * K.log(pred_bkg)
    lossobj = -250 * true_obj * K.log(pred_obj)
    loss = K.sum(K.concatenate([lossbg, lossobj]), -1)
    return loss


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