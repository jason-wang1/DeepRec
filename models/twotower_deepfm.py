# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1, l2
from layers.tower_deepfm import TowerDeepFM
from layers.input_to_wide_emb import InputToWideEmb


class TwoTowerDeepFM(Model):
    def __init__(self, config, **kwargs):
        user_feat_list = config["model_config"]["feature_groups"]["user"]
        item_feat_list = config["model_config"]["feature_groups"]["item"]
        self.user_feat_config = [e for e in config["feature_config"]["features"] if e["input_names"] in user_feat_list]
        self.item_feat_config = [e for e in config["feature_config"]["features"] if e["input_names"] in item_feat_list]
        self.emb_dim = config["model_config"]["embedding_dim"]
        self.dnn_shape = config["model_config"]["deep_hidden_units"]
        if "l2_reg" in config["model_config"]:
            self.reg = l2(config["model_config"]["l2_reg"])
        elif "l1_reg" in config["model_config"]:
            self.reg = l1(config["model_config"]["l1_reg"])
        else:
            self.reg = None
        super(TwoTowerDeepFM, self).__init__(**kwargs)  # Be sure to call this somewhere!

    def build(self, input_shape):
        self.user_input_to_wide_emb = InputToWideEmb(self.emb_dim, self.user_feat_config, self.reg, name="user_input")
        self.item_input_to_wide_emb = InputToWideEmb(self.emb_dim, self.item_feat_config, self.reg, name="item_input")
        self.user_tower = TowerDeepFM("user", self.dnn_shape, self.reg, name="user_tower")
        self.item_tower = TowerDeepFM("item", self.dnn_shape, self.reg, name="item_tower")

    def call(self, inputs, training=None, mask=None):
        user_wide_input, user_fm_input = self.user_input_to_wide_emb(inputs)
        item_wide_input, item_fm_input = self.item_input_to_wide_emb(inputs)

        user_represent = self.user_tower({"wide_input": user_wide_input, "fm_input": user_fm_input})  # (batch_size, represent_size)
        item_represent = self.item_tower({"wide_input": item_wide_input, "fm_input": item_fm_input})  # (batch_size, represent_size)
        pred = tf.reduce_sum(user_represent * item_represent, axis=1, keepdims=True)
        pred = tf.sigmoid(pred)
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
