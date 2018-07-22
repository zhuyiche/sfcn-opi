import numpy as np
import scipy
import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer

class DENSECRF(Layer):
    def __int__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DENSECRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='cosine_distance_weight',
                                      initializer='uniform',
                                      trainable=True)
    def call(self, inputs, **kwargs):
        """
        Input is a list of [feats, logits, iteration]
        feats for now is shape [batch_size, num_nodes, channel]
        logits for now is shape [batch_size, num_nodes, channel]
        :param inputs:
        :param kwargs:
        :return:
        """
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('Merge must be called on a list of tensors '
                            'Got: ' + str(inputs))
        feats, logits = inputs[0], inputs[1]
        iteration = inputs[2]

        feats_norm = np.linalg.norm(feats, ord=2, axis=2, keepdims=True)
        print('feats_norm.shape: ', feats_norm.shape)
        pairwise_norm = np.dot(feats_norm, np.transpose(feats_norm, [1, 2]))
        print('pairwise_norm.shape: ', pairwise_norm.shape)
        pairwise_dot = np.dot(feats, np.transpose(feats, [1, 2]))
        print('pairwise_dot.shape: ', pairwise_dot)
        pairwise_sim = pairwise_dot/pairwise_norm
        w_sym = (self.w + np.transpose(self.w, [1,2])) /2
        pairwise_potential = pairwise_sim * w_sym

        for i in range(iteration):
            probs = tf.sigmoid(logits[])
            pairwise_potential_E = np.sum(probs * pairwise_potential - (1 - probs) * pairwise_potential)
            logits = unary_potenial + pairwise_potential_E

        return logits

    def compute_output_shape(self, input_shape):


