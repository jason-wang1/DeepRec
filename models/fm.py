# -*- coding:utf-8 -*-
import tensorflow as tf
from models.base import Base
from layers.input_to_wide_emb import InputToWideEmb
from layers.fm import FMLayer


class FM(Base):
    def __init__(self, config, **kwargs):
        super(FM, self).__init__(config, **kwargs)

    def build(self, input_shape):
        assert len(self.feature_group_list) == 1
        self.input_to_wide_emb = InputToWideEmb(True, self.emb_dim, self.feature_group_list[0][1], self.input_attributes, self.reg, name=self.feature_group_list[0][0])
        self.fm = FMLayer(name="fm_layer")
        self.bias = self.add_weight(name='bias', shape=[1], initializer='zeros')

    def call(self, inputs, training=None, mask=None):
        wide_input, fm_input = self.input_to_wide_emb(inputs)
        wide_output = tf.reduce_sum(wide_input, axis=1,  keepdims=True) + self.bias  # (batch_size, 1)
        fm_output = self.fm(fm_input)  # (batch_size, 1)
        output = tf.sigmoid(wide_output + fm_output)
        return output
