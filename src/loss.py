import keras.backend as K
import tensorflow as tf
import numpy as np

epsilon = 1e-7
cls_threshold = 0.8


def detection_loss_weight(weight):
    def detection_loss(y_true, y_pred):
        logits = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        targets = tf.convert_to_tensor(y_true, dtype=tf.float32)
        try:
            targets.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError(
                "logits and targets must have the same shape (%s vs %s)" %
                (logits.get_shape(), targets.get_shape()))
        left = weight * tf.log(y_pred)
        right = (1-y_true) * tf.log(1-y_pred)
        result = -tf.reduce_mean(tf.add(left, right))
        return result
    return detection_loss


def classification_loss_weight(weights, threshold=cls_threshold):
    def classification_loss(y_true, y_pred):
        indicator = tf.cast(tf.greater(y_true, threshold), tf.uint8)
        class_weights = tf.convert_to_tensor(weights, dtype=tf.uint8)
        try:
            y_pred.get_shape().assert_is_compatible_with(indicator.get_shape())
        except ValueError:
            raise ValueError(
                 "indicator must have the same shape (%s vs %s)" %
                 (indicator.get_shape(), y_pred.get_shape())
                            )
        loss = -tf.multiply(class_weights, tf.multiply(indicator, tf.log(y_pred)))
        return loss
    return classification_loss


def joint_loss_weight(det_weights, cls_joint_weights, joint_weights, cls_threshold = cls_threshold):
    def joint_loss(y_true, y_pred):
        def detection_loss(y_true, y_pred, det_weights):
            logits = tf.convert_to_tensor(y_pred, dtype=tf.float16)
            targets = tf.convert_to_tensor(y_true, dtype=tf.float16)
            try:
                targets.get_shape().merge_with(logits.get_shape())
            except ValueError:
                raise ValueError(
                    "logits and targets must have the same shape (%s vs %s)" %
                    (logits.get_shape(), targets.get_shape()))
            left = det_weights * tf.log(y_pred)
            right = (1 - targets) * tf.log(1 - y_pred)
            result = -tf.reduce_mean(tf.add(left, right))
            return result

        def classification_loss(y_true, y_pred, cls_joint_weights, threshold):
            indicator = tf.cast(tf.greater(y_true, cls_threshold), tf.uint8)
            class_weights = tf.convert_to_tensor(joint_weights, dtype=tf.uint8)
            try:
                y_pred.get_shape().assert_is_compatible_with(indicator.get_shape())
            except ValueError:
                raise ValueError(
                    "indicator must have the same shape (%s vs %s)" %
                    (indicator.get_shape(), y_pred.get_shape())
                )
            loss = -tf.multiply(class_weights, tf.multiply(indicator, tf.log(y_pred)))
            return loss

        det_loss = detection_loss(y_true, y_pred, det_weights)
        cls_loss = classification_loss(y_true, y_pred, cls_joint_weights, cls_threshold)
        total_loss = tf.add(det_loss, tf.multiply(cls_loss, joint_weights))
        return total_loss
    return joint_loss


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