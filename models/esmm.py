from models.base import Base
import tensorflow as tf
from layers.dnn import DNN
from layers.input_to_wide_emb import InputToWideEmb


class ESMM(Base):
    def __init__(self, config, **kwargs):
        self.dnn_shape = config["model_config"]["deep_hidden_units"]
        super(ESMM, self).__init__(config, **kwargs)

    def build(self, input_shape):
        self.input_to_wide_emb_list = [InputToWideEmb(False, self.emb_dim, group, self.input_attributes, self.reg, name=group_name) for group_name, group in self.feature_group_list]
        self.ctr_tower = DNN(dnn_shape=self.dnn_shape, reg=self.reg, name="ctr_tower")
        self.cvr_tower = DNN(dnn_shape=self.dnn_shape, reg=self.reg, name="cvr_tower")

    def call(self, inputs, training=None, mask=None):
        emb_tensor_list = []
        for input_to_wide_emb in self.input_to_wide_emb_list:
            emb_tensor = tf.reduce_sum(input_to_wide_emb(inputs), axis=1)  # (batch_size, emb_size)
            emb_tensor_list.append(emb_tensor)
        dnn_input = tf.concat(emb_tensor_list, axis=1)  # (batch_size, feat_group_size * emb_size)
        ctr_pred = self.ctr_tower(dnn_input)  # (batch_size, 1)
        ctr_pred = tf.squeeze(tf.sigmoid(ctr_pred), axis=1)
        cvr_pred = self.cvr_tower(dnn_input)  # (batch_size, 1)
        cvr_pred = tf.squeeze(tf.sigmoid(cvr_pred), axis=1)
        ctcvr_pred = ctr_pred * cvr_pred
        return ctr_pred, ctcvr_pred
