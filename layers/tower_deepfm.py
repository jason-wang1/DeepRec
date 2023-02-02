import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from layers.fm import FMLayer
from layers.dnn import DNN
import sys

class TowerDeepFM(Layer):
    """
      Input shape
        - a tuple: (wide_tensor, emb_tensor)
        wide_tensor shape ``(batch_size, feat_size)``
        emb_tensor  shape ``(batch_size, feat_size, emb_size)``

      Output shape
        - ``(batch_size, represent_vector_size)``
    """
    def __init__(self, feature_group, feat_size, emb_dim, dnn_shape, reg, **kwargs):
        self.feature_group = feature_group
        self.feat_size = feat_size
        self.emb_dim = emb_dim
        self.dnn_shape = dnn_shape
        self.reg = reg
        super(TowerDeepFM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.fm = FMLayer(name="fm")
        self.dnn = DNN(dnn_shape=self.dnn_shape, reg=self.reg, name="dnn")
        super(TowerDeepFM, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        wide_input = inputs["wide_input"]  # (batch_size, feat_size)
        fm_input = inputs["fm_input"]  # (batch_size, feat_size, emb_size)

        wide_output = tf.reduce_sum(wide_input, axis=1, keepdims=True)  # (batch_size, 1)

        fm_output = self.fm(fm_input)  # (batch_size, 1)
        emb_vec_output = tf.reduce_sum(fm_input, axis=1)  # (batch_size, emb_size)

        dnn_input = tf.reshape(fm_input, [-1, self.feat_size * self.emb_dim])
        dnn_output = self.dnn(dnn_input)  # (batch_size, user_deep_vector_size)

        wide_ones = tf.ones_like(wide_output)
        fm_ones = tf.ones_like(fm_output)
        if self.feature_group == "user":
            res = tf.concat([wide_output, fm_output, emb_vec_output, dnn_output, wide_ones, fm_ones], axis=1)
        elif self.feature_group == "item":
            res = tf.concat([wide_ones, fm_ones, emb_vec_output, dnn_output, wide_output, fm_output], axis=1)
        else:
            print(f"error feature_group: {self.feature_group}")
            sys.exit(1)
        return res
