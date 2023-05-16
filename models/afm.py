# -*- coding:utf-8 -*-
import tensorflow as tf
from models.base import Base
from layers.input_to_wide_emb import InputToWideEmb
from layers.afm import AFMLayer


class AFM(Base):
    def __init__(self, config, **kwargs):
        super(AFM, self).__init__(config, **kwargs)

    def build(self, input_shape):
        self.input_to_wide_emb_list = [InputToWideEmb(True, self.emb_dim, group, self.input_attributes, self.reg, name=group_name) for group_name, group in self.feature_group_list]
        feat_size = 0
        for group_name, group in self.feature_group_list:
            feat_size += len(group)
        print(f"feat_size: {feat_size}")
        self.afm = AFMLayer(4, feat_size, self.reg, name="afm_layer")
        self.bias = self.add_weight(name='bias', shape=[1], initializer='zeros')

    def call(self, inputs, training=None, mask=None):
        emb_tensor_list = []
        wide_tensor_list = []
        for input_to_wide_emb in self.input_to_wide_emb_list:
            wide_tensor, emb_tensor = input_to_wide_emb(inputs)
            wide_tensor_list.append(wide_tensor)
            emb_tensor_list.append(emb_tensor)
        wide_input = tf.concat(wide_tensor_list, axis=1)
        wide_output = tf.reduce_sum(wide_input, axis=1, keepdims=True) + self.bias  # (batch_size, 1)
        afm_input = tf.concat(emb_tensor_list, axis=1)
        fm_output = self.afm(afm_input)  # (batch_size, 1)
        output = tf.sigmoid(wide_output + fm_output)
        return output
