import keras.backend as K
import tensorflow as tf
import numpy as np

epsilon = 1e-7
cls_threshold = 0.8



def detection_loss(weight):
    def _detection_loss(y_true, y_pred):
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weight))
    return _detection_loss


def classification_loss(weights, smooth_factor, threshold=cls_threshold):

    def _classification_loss(y_true, y_pred):
        indicator = tf.greater_equal(y_pred, threshold, name='indicator_great')
        indicator = tf.cast(indicator, tf.float32, name='indicator_cast')
        class_weights = tf.convert_to_tensor(weights, name='cls_weight_convert')
        class_weights = tf.cast(class_weights, tf.float32)
        # logits = tf.convert_to_tensor(y_pred, name='logits_convert', dtype=tf.float64)
        logits = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        logits = tf.cast(logits, tf.float32, name='logits_cast')
        smooth_factors = tf.cast(smooth_factor, tf.float32)
        try:
            y_pred.get_shape().assert_is_compatible_with(indicator.get_shape())
        except ValueError:
            raise ValueError(
                 "indicator must have the same shape (%s vs %s)" %
                 (indicator.get_shape(), y_pred.get_shape())
                            )
        loss = - class_weights * indicator * tf.log(y_pred) * tf.pow((1-y_pred), smooth_factors)
        return loss

    return _classification_loss


def joint_loss(det_weights, cls_joint_weights, joint_weights, smooth_factor, cls_threshold = cls_threshold):
    def _joint_loss(y_true, y_pred):
        def detection_loss(y_true, y_pred, det_weights):
            return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, det_weights))

        def classification_loss(y_true, y_pred, cls_joint_weights, threshold):
            indicator = tf.cast(tf.greater_equal(y_true, threshold), tf.float32)
            class_weights = tf.convert_to_tensor(cls_joint_weights)
            try:
                y_pred.get_shape().assert_is_compatible_with(indicator.get_shape())
            except ValueError:
                raise ValueError(
                    "indicator must have the same shape (%s vs %s)" %
                    (indicator.get_shape(), y_pred.get_shape())
                )
            loss = - class_weights * indicator * tf.log(y_pred) * tf.pow((1-y_pred), smooth_factor)
            return loss

        det_loss = detection_loss(y_true, y_pred, det_weights)
        cls_loss = classification_loss(y_true, y_pred, cls_joint_weights, cls_threshold)
        total_loss = tf.add(det_loss, tf.multiply(cls_loss, joint_weights))
        return total_loss
    return _joint_loss


"""
def detection_loss_weight(weight):
    def detection_loss(y_true, y_pred):
        logits = tf.convert_to_tensor(y_pred)
        targets = tf.convert_to_tensor(y_true)
        weights = tf.convert_to_tensor(weight)
        try:
            targets.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError(
                "logits and targets must have the same shape (%s vs %s)" %
                (logits.get_shape(), targets.get_shape()))
        left = weights * tf.log(logits)
        right = (1 - targets) * tf.log(1 - logits)
        result = -tf.reduce_mean(tf.add(left, right))
        return result
    return detection_loss
"""

"""
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
"""
"""
        true_bkg = y_true[:, :, :, 0:1]
        pred_bkg = K.clip(y_pred[:, :, :, 0:1], epsilon, 1 - epsilon)
        true_obj = y_true[:, :, :, 1:2]
        pred_obj = K.clip(y_pred[:, :, :, 1:2], epsilon, 1 - epsilon)
        lossbg = - true_bkg * K.log(pred_bkg)
        lossobj = -250 * true_obj * K.log(pred_obj)
        loss = K.sum(K.concatenate([lossbg, lossobj]), -1)
        """