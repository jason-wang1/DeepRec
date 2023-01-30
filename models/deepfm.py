# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1, l2
from layers.fm import FM
from layers.dnn import DNN
from layers.input_to_wide_emb import InputToWideEmb


class DeepFM(Model):
    def __init__(self, config, **kwargs):
        self.config = config
        self.feat_size = len(config["feature_config"]["features"])
        self.emb_dim = config["model_config"]["embedding_dim"]
        self.dnn_shape = config["model_config"]["deep_hidden_units"]
        if "l2_reg" in config["model_config"]:
            self.reg = l2(config["model_config"]["l2_reg"])
        elif "l1_reg" in config["model_config"]:
            self.reg = l1(config["model_config"]["l1_reg"])
        else:
            self.reg = None
        super(DeepFM, self).__init__(**kwargs)  # Be sure to call this somewhere!

    def build(self, input_shape):
        self.input_to_wide_emb = InputToWideEmb(self.emb_dim, self.config["feature_config"]["features"], self.reg)
        self.fm = FM(name="fm")
        self.dnn = DNN(dnn_shape=self.dnn_shape, reg=self.reg, name="dnn")
        self.bias = self.add_weight(name='bias', shape=[1], initializer='zeros')

    def call(self, inputs, training=None, mask=None):
        wide_input, fm_input = self.input_to_wide_emb(inputs)
        wide_output = tf.reduce_sum(wide_input, axis=1,  keepdims=True) + self.bias  # (batch_size, 1)
        fm_output = self.fm(fm_input)  # (batch_size, 1)
        dnn_input = tf.reshape(fm_input, [-1, self.feat_size*self.emb_dim])
        dnn_output = self.dnn(dnn_input)  # (batch_size, 1)

        output = tf.sigmoid(wide_output + fm_output + dnn_output)
        return output
