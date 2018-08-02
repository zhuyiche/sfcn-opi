import keras.backend as K
import tensorflow as tf
import numpy as np
epsilon = 1e-7
cls_threshold = 0.8




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
            loss = -tf.reduce_mean(class_weights * indicator * tf.log(logits, name='logitslog'))
            return loss

        det_loss = _detection_loss(y_true, y_pred, det_weights)
        cls_loss = _classification_loss(y_true, y_pred, cls_joint_weights, cls_threshold)
        total_loss = tf.add(det_loss, tf.multiply(cls_loss, joint_weights))
        return total_loss
    return _joint_loss