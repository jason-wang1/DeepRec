import tensorflow as tf
from models.base import Base
from layers.tower_deepfm import TowerDeepFM
from layers.input_to_wide_emb import InputToWideEmb


class TwoTowerDeepFM(Base):
    def __init__(self, config, **kwargs):
        self.dnn_shape = config["model_config"]["deep_hidden_units"]
        super(TwoTowerDeepFM, self).__init__(config, **kwargs)

    def build(self, input_shape):
        assert len(self.config["model_config"]["feature_groups"]) == 2
        self.user_input_to_wide_emb = InputToWideEmb(True, self.emb_dim, self.config["model_config"]["feature_groups"]["user"], self.input_attributes, self.reg, name="user_input")
        self.item_input_to_wide_emb = InputToWideEmb(True, self.emb_dim, self.config["model_config"]["feature_groups"]["item"], self.input_attributes, self.reg, name="item_input")
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
