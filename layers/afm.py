import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Dense
import itertools


class AFMLayer(Layer):
    """Attentonal Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size, feat_size, embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
    """
    def __init__(self, attention_factor, feat_size, reg, **kwargs):
        self.attention_factor = attention_factor
        self.feat_size = feat_size
        self.reg = reg
        super(AFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"A `AFM` layer requires inputs of a list with same shape tensor like "
                             f"(None, feat_size, embedding_size) Got different shapes: {input_shape}")
        self.W_dense = Dense(self.attention_factor, activation='relu', kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.h_dense = Dense(1, activation=None, use_bias=False, kernel_regularizer=self.reg)
        self.p_dense = Dense(1, activation=None, use_bias=False, kernel_regularizer=self.reg)
        super(AFMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs: (batch_size, feat_size, embedding_size)
        """
        input_list = tf.split(inputs, num_or_size_splits=self.feat_size, axis=1)  # [(batch_size, 1, emb_size)]
        p = []
        q = []
        for f1, f2 in itertools.combinations(input_list, 2):
            p.append(f1)
            q.append(f2)
        p = tf.concat(p, axis=1)
        q = tf.concat(q, axis=1)
        bi_interaction = p * q  # (batch_size, bi_interaction_size, emb_size)
        att_tensor = self.W_dense(bi_interaction)  # (batch_size, bi_interaction_size, attention_factor)
        att_tensor = self.h_dense(att_tensor)  # (batch_size, bi_interaction_size, 1)
        att_tensor = tf.nn.softmax(att_tensor, axis=1) * (((self.feat_size-1) * self.feat_size)/2.0)  # (batch_size, bi_interaction_size, 1)
        att_bi_interaction = tf.reduce_sum(bi_interaction * att_tensor, axis=1)  # (batch_size, emb_size)
        output = self.p_dense(att_bi_interaction)
        return output
