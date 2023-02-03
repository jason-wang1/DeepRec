import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1, l2
from layers.dnn import DNN
from layers.input_to_wide_emb import InputToWideEmb

class ESMM(Model):
    def __init__(self, config, **kwargs):
        self.config = config
        self.feat_size = len(config["feature_config"]["features"])
        self.emb_dim = config["model_config"]["embedding_dim"]
        self.dnn_shape = config["model_config"]["deep_hidden_units"]
        if self.dnn_shape[-1] != 1:
            raise ValueError(f"unexpected dnn_shape: {self.dnn_shape}")
        if "l2_reg" in config["model_config"]:
            self.reg = l2(config["model_config"]["l2_reg"])
        elif "l1_reg" in config["model_config"]:
            self.reg = l1(config["model_config"]["l1_reg"])
        else:
            self.reg = None
        super(ESMM, self).__init__(**kwargs)  # Be sure to call this somewhere!

    def build(self, input_shape):
        self.user_input_to_wide_emb = InputToWideEmb(self.emb_dim, self.config["feature_config"]["features"],
                                                     self.reg, name="user_input")
        self.item_input_to_wide_emb = InputToWideEmb(self.emb_dim, self.config["feature_config"]["features"],
                                                     self.reg, name="item_input")
        self.ctr_tower = DNN(dnn_shape=self.dnn_shape, reg=self.reg, name="ctr_tower")
        self.cvr_tower = DNN(dnn_shape=self.dnn_shape, reg=self.reg, name="cvr_tower")

    def call(self, inputs, training=None, mask=None):
        user_wide_tensor, user_emb_tensor = self.user_input_to_wide_emb(inputs)
        item_wide_tensor, item_emb_tensor = self.item_input_to_wide_emb(inputs)
        user = tf.reduce_sum(user_emb_tensor, axis=1)  # (batch_size, emb_size)
        item = tf.reduce_sum(item_emb_tensor, axis=1)  # (batch_size, emb_size)
        dnn_input = tf.concat([user, item], axis=1)  # (batch_size, 2 * emb_size)
        ctr_pred = self.ctr_tower(dnn_input)  # (batch_size, 1)
        cvr_pred = self.cvr_tower(dnn_input)  # (batch_size, 1)
        ctcvr_pred = ctr_pred * cvr_pred
        return ctr_pred, ctcvr_pred
