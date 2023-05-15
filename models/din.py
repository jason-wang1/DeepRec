# -*- coding:utf-8 -*-
import tensorflow as tf
from models.base import Base
from layers.tower_deepfm import DNN
from layers.input_to_wide_emb import InputToWideEmb, AttentionSequencePoolingInput
from tensorflow.python.keras.layers import Flatten


class DIN(Base):
    def __init__(self, config, **kwargs):
        self.feature_group_list = [(group_name + "_input", group) for group_name, group in config["model_config"]["feature_groups"].items() if group_name != "din"]
        self.item_group_index = 0
        for i, group_name in enumerate(config["model_config"]["feature_groups"].keys()):
            if group_name == "item":
                self.item_group_index = i
        super(DIN, self).__init__(config, **kwargs)  # Be sure to call this somewhere!

    def build(self, input_shape):
        item_feat_group = self.config["model_config"]["feature_groups"]["item"]
        din_feat_group_list = self.config["model_config"]["feature_groups"]["din"]
        self.item_gather_list = []
        for din_feat_group in din_feat_group_list:
            gather = []
            for din_feat in din_feat_group:
                cross_item = din_feat["cross_item"]
                for i, item_feat in enumerate(item_feat_group):
                    if cross_item == item_feat["input_names"]:
                        gather.append(i)
            assert len(gather) == len(din_feat_group)
            self.item_gather_list.append(gather)
        self.input_to_wide_emb_list = [
            InputToWideEmb(False, self.emb_dim, group, self.input_attributes, self.reg, name=group_name) for
            group_name, group in self.feature_group_list]
        din_dnn_shape = self.config["model_config"]["din_hidden_units"]
        self.att_layer_list = [AttentionSequencePoolingInput(din_feat_group, self.input_attributes, self.emb_dim, self.reg, din_dnn_shape, name="din_input")
                               for din_feat_group in din_feat_group_list]
        self.flatten = Flatten()
        final_dnn_shape = self.config["model_config"]["deep_hidden_units"]
        self.dnn = DNN(final_dnn_shape, self.reg, final_activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        dnn_input_list = [input_to_wide_emb(inputs) for input_to_wide_emb in self.input_to_wide_emb_list]  # (batch_size, feat_size, emb_dim)
        for item_gather, att_layer in zip(self.item_gather_list, self.att_layer_list):
            query_emb = tf.gather(dnn_input_list[self.item_group_index], indices=item_gather, axis=1)
            query_emb = tf.reshape(query_emb, shape=[self.batch_size, 1, len(item_gather) * self.emb_dim])
            seq_emb_input = att_layer([query_emb, inputs])  # (batch_size, 1, m * emb_dim)
            dnn_input_list.append(seq_emb_input)
        dnn_input = tf.concat(dnn_input_list, axis=1)  # (batch_size, feat_size*group_size, emb_dim)
        dnn_input = self.flatten(dnn_input)
        pred = self.dnn(dnn_input)
        return pred
