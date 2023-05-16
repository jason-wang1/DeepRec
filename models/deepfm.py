# -*- coding:utf-8 -*-
import tensorflow as tf

from layers.dnn import DNN
from layers.fm import FMLayer
from layers.input_to_wide_emb import InputToWideEmb
from models.base import Base
from tensorflow.python.keras.layers import Flatten


class DeepFM(Base):
    def __init__(self, config, **kwargs):
        self.dnn_shape = config["model_config"]["deep_hidden_units"]
        super(DeepFM, self).__init__(config, **kwargs)  # Be sure to call this somewhere!

    def build(self, input_shape):
        assert len(self.feature_group_list) == 1
        self.input_to_wide_emb = InputToWideEmb(True, self.emb_dim, self.feature_group_list[0][1], self.input_attributes, self.reg, name=self.feature_group_list[0][0])
        self.flatten = Flatten()
        self.fm = FMLayer(name="fm")
        self.dnn = DNN(dnn_shape=self.dnn_shape, reg=self.reg, name="dnn")
        self.bias = self.add_weight(name='bias', shape=[1], initializer='zeros')

    def call(self, inputs, training=None, mask=None):
        wide_input, fm_input = self.input_to_wide_emb(inputs)
        wide_output = tf.reduce_sum(wide_input, axis=1,  keepdims=True) + self.bias  # (batch_size, 1)
        fm_output = self.fm(fm_input)  # (batch_size, 1)
        dnn_input = self.flatten(fm_input)  # (batch_size, feat_size*emb_dim)
        dnn_output = self.dnn(dnn_input)  # (batch_size, 1)

        output = tf.sigmoid(wide_output + fm_output + dnn_output)
        return output
