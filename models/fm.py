# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1, l2
from layers.input_to_wide_emb import InputToWideEmb
from layers.fm import FMLayer


class FM(Model):
    def __init__(self, config, **kwargs):
        self.config = config
        self.feat_size = len(config["feature_config"]["features"])
        self.emb_dim = config["model_config"]["embedding_dim"]
        if "l2_reg" in config["model_config"]:
            self.reg = l2(config["model_config"]["l2_reg"])
        elif "l1_reg" in config["model_config"]:
            self.reg = l1(config["model_config"]["l1_reg"])
        else:
            self.reg = None
        super(FM, self).__init__(**kwargs)  # Be sure to call this somewhere!

    def build(self, input_shape):
        self.input_to_wide_emb = InputToWideEmb(self.emb_dim, self.config["feature_config"]["features"], self.reg)
        self.fm = FMLayer(name="fm_layer")
        self.bias = self.add_weight(name='bias', shape=[1], initializer='zeros')

    def call(self, inputs, training=None, mask=None):
        wide_input, afm_input = self.input_to_wide_emb(inputs)
        wide_output = tf.reduce_sum(wide_input, axis=1,  keepdims=True) + self.bias  # (batch_size, 1)
        fm_output = self.fm(afm_input)  # (batch_size, 1)
        output = tf.sigmoid(wide_output + fm_output)
        return output
