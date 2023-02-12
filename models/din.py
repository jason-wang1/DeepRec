# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1, l2
from layers.tower_deepfm import DNN
from layers.input_to_wide_emb import InputToWideEmb, AttentionSequencePoolingInput
from tensorflow.python.keras.layers import Flatten

class DIN(Model):
    def __init__(self, config, **kwargs):
        user_feat_list = config["model_config"]["feature_groups"]["user"]
        item_feat_list = config["model_config"]["feature_groups"]["item"]
        self.din_feat_list = config["model_config"]["feature_groups"]["din"]
        self.din_feat_num = len(self.din_feat_list)
        self.pad_num = config["feature_config"]["pad_num"]
        self.user_feat_config = [e for e in config["feature_config"]["features"] if e["input_names"] in user_feat_list]
        self.item_feat_config = [e for e in config["feature_config"]["features"] if e["input_names"] in item_feat_list]
        self.din_feat_config_list = [e for e in config["feature_config"]["features"] if e["input_names"] in self.din_feat_list]
        self.emb_dim = config["model_config"]["embedding_dim"]
        self.final_dnn_shape = config["model_config"]["deep_hidden_units"]
        self.din_dnn_shape = config["model_config"]["din_hidden_units"]
        if "l2_reg" in config["model_config"]:
            self.reg = l2(config["model_config"]["l2_reg"])
        elif "l1_reg" in config["model_config"]:
            self.reg = l1(config["model_config"]["l1_reg"])
        else:
            self.reg = None
        super(DIN, self).__init__(**kwargs)  # Be sure to call this somewhere!

    def build(self, input_shape):
        self.user_input_to_wide_emb = InputToWideEmb(False, self.emb_dim, self.user_feat_config, self.reg, name="user_input")
        self.item_input_to_wide_emb = InputToWideEmb(False, self.emb_dim, self.item_feat_config, self.reg, name="item_input")
        self.att_layer = AttentionSequencePoolingInput(self.din_feat_config_list, self.emb_dim, self.reg,
                                                       self.din_dnn_shape, pad_num=self.pad_num, name="din_input")
        self.flatten = Flatten()
        self.dnn = DNN(self.final_dnn_shape, self.reg, final_activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        user_emb_input = self.user_input_to_wide_emb(inputs)  # (batch_size, feat_size, emb_dim)
        item_emb_input = self.item_input_to_wide_emb(inputs)  # (batch_size, feat_size, emb_dim)
        query_emb = tf.slice(item_emb_input, begin=[0, 0, 0], size=[-1, self.din_feat_num, -1])
        query_emb = tf.reshape(query_emb, shape=[-1, 1, self.din_feat_num * self.emb_dim])
        keys_length = inputs[self.din_feat_list[0]].row_lengths(axis=1)  # (batch_size,)
        keys = [inputs[key].to_tensor(default_value=0, shape=[None, self.pad_num]) for key in self.din_feat_list]
        seq_emb_input = self.att_layer([query_emb, keys, keys_length])  # (batch_size, 1, m * emb_dim)
        dnn_input = tf.concat([self.flatten(user_emb_input), self.flatten(seq_emb_input), self.flatten(item_emb_input)], axis=-1)
        pred = self.dnn(dnn_input)
        return pred

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
