import keras.backend as K
import tensorflow as tf
import numpy as np
epsilon = 1e-7
cls_threshold = 0.8


def deeplab_cls_loss(weights):
    def _cls_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1-epsilon)
        y_pred_cls1 = y_pred[:, :, :, 1]
        y_pred_cls2 = y_pred[:, :, :, 2]
        y_pred_cls3 = y_pred[:, :, :, 3]
        y_pred_cls4 = y_pred[:, :, :, 4]
        y_true_cls1 = y_true[:, :, :, 1]
        y_true_cls2 = y_true[:, :, :, 2]
        y_true_cls3 = y_true[:, :, :, 3]
        y_true_cls4 = y_true[:, :, :, 4]



def detection_double_focal_loss_K(weight, fkg_focal_smoother, bkg_focal_smoother):
    """
    Binary crossentropy loss with focal.
    :param weight:
    :param focal_smoother:
    :return:
    """
    def _detection_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = y_pred[:, :, :, 1]
        y_true = y_true[:, :, :, 1]
        fkg_smooth = K.pow((1-y_pred), fkg_focal_smoother)
        bkg_smooth = K.pow(y_pred, bkg_focal_smoother)
        result = -K.mean(weight[1] * fkg_smooth * y_true * K.log(y_pred) +
                         weight[0] * bkg_smooth * (1-y_true) * K.log(1-y_pred))
        return result
    return _detection_loss


def detection_double_focal_loss_indicator_K(weight, fkg_focal_smoother,
                                            bkg_focal_smoother,
                                            indicator_weight):
    """
    Binary crossentropy loss with focal.
    :param weight:
    :param focal_smoother:
    :return:
    """
    def _detection_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = y_pred[:, :, :, 1]
        y_true = y_true[:, :, :, 1]
        fkg_smooth = K.pow((1-y_pred), fkg_focal_smoother)
        bkg_smooth = K.pow(y_pred, bkg_focal_smoother)
        indicator = K.greater_equal(y_pred, cls_threshold)
        result = -K.mean((fkg_smooth + indicator_weight * indicator) * weight[1] * y_true * K.log(y_pred) +
                                   bkg_smooth * (1-y_true) * K.log(1-y_pred) * weight[0])
        return result
    return _detection_loss


def detection_loss_without_part2_K(weight):
    """
    Crossentropy loss without focal.
    :param weight:
    :return:
    """
    def _detection_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        result = -K.mean(weight * (y_true * K.log(y_pred)))
        return result
    return _detection_loss


def detection_loss_K(weight):
    """
    Binary crossentropy loss without focal
    :param weight:
    :return:
    """
    def _detection_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = y_pred[:, :, :, 1]
        y_true = y_true[:, :, :, 1]
        result = -K.mean(weight[1] * y_true * K.log(y_pred) +
                         weight[0] * (1-y_true) * K.log(1-y_pred))
        return result
    return _detection_loss


def detection_focal_loss_K(weight, focal_smoother):
    """
    Binary crossentropy loss with focal.
    :param weight:
    :param focal_smoother:
    :return:
    """
    def _detection_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        smooth = K.pow((1-y_pred), focal_smoother)
        result = -K.mean(weight * (smooth * y_true * K.log(y_pred) + (1-y_true) * K.log(1-y_pred)))
        return result
    return _detection_loss


def detection_focal_loss_without_part2_K(weight, focal_smoother):
    """

    :param weight:
    :param focal_smoother:
    :return:
    """
    def _detection_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        smooth = K.pow((1-y_pred), focal_smoother)
        result = -K.mean(weight * (smooth * y_true * K.log(y_pred)))
        return result
    return _detection_loss


def detection_loss(weight):
    def _detection_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1-epsilon)
        weights = tf.convert_to_tensor(weight)
        weights = tf.cast(weights, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        result = -tf.reduce_mean(weights * y_true * tf.log(y_pred) + (1-y_true) * tf.log(1-y_pred))
        return result
    return _detection_loss


def classification_loss(weights, threshold=cls_threshold):

    def _classification_loss(y_true, y_pred):
        indicator = tf.greater_equal(y_pred, threshold, name='indicator_great')
        indicator = tf.cast(indicator, tf.float32, name='indicator_cast')
        class_weights = tf.convert_to_tensor(weights, name='cls_weight_convert')
        class_weights = tf.cast(class_weights,tf.float32)
        # logits = tf.convert_to_tensor(y_pred, name='logits_convert', dtype=tf.float64)
        logits = tf.clip_by_value(y_pred, epsilon, 1-epsilon)
        logits = tf.cast(logits, tf.float32, name='logits_cast')
        """
        try:
            y_pred.get_shape().assert_is_compatible_with(indicator.get_shape())
        except ValueError:
            raise ValueError(
                 "indicator must have the same shape (%s vs %s)" %
                 (indicator.get_shape(), y_pred.get_shape()))
        """
        loss = -tf.reduce_mean(class_weights * indicator * tf.log(logits, name='logitslog'))
        return loss

    return _classification_loss


def joint_loss(det_weights, cls_joint_weights, joint_weights, cls_threshold = cls_threshold):
    def _joint_loss(y_true, y_pred):
        def _detection_loss(y_true, y_pred, det_weights):
            return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, det_weights))

        def _classification_loss(y_true, y_pred, cls_joint_weights, threshold):
            indicator = tf.greater_equal(y_pred, threshold, name='indicator_great')
            indicator = tf.cast(indicator, tf.float32, name='indicator_cast')
            class_weights = tf.convert_to_tensor(cls_joint_weights, name='cls_weight_convert')
            class_weights = tf.cast(class_weights, tf.float32)
            # logits = tf.convert_to_tensor(y_pred, name='logits_convert', dtype=tf.float64)
            logits = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
            logits = tf.cast(logits, tf.float32, name='logits_cast')
            """
            try:
                y_pred.get_shape().assert_is_compatible_with(indicator.get_shape())
            except ValueError:
                raise ValueError(
                     "indicator must have the same shape (%s vs %s)" %
                     (indicator.get_shape(), y_pred.get_shape())
                                )
                                """
            loss = -tf.reduce_mean(class_weights * indicator * tf.log(logits, name='logitslog'))
            return loss

        det_loss = _detection_loss(y_true, y_pred, det_weights)
        cls_loss = _classification_loss(y_true, y_pred, cls_joint_weights, cls_threshold)
        total_loss = tf.add(det_loss, tf.multiply(cls_loss, joint_weights))
        return total_loss
    return _joint_loss


def detection_focal_loss(weight, focal_smoother):
    def _detection_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1-epsilon)
        weights = tf.convert_to_tensor(weight)
        weights = tf.cast(weights, tf.float32)
        focal_smooth = tf.cast(focal_smoother, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        smooth = tf.pow((1-y_pred), focal_smooth)
        result = -tf.reduce_mean(weights * (smooth * y_true * tf.log(y_pred) + (1-y_true) * tf.log(1-y_pred)))
        return result
    return _detection_loss

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