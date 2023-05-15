# -*- coding:utf-8 -*-
from layers.din_tower import DinTower
from layers.input_to_wide_emb import InputToWideEmb
from models.base import Base


class DIN(Base):
    def __init__(self, config, **kwargs):
        super(DIN, self).__init__(config, **kwargs)

    def build(self, input_shape):
        input_to_wide_emb_list = [
            InputToWideEmb(False, self.emb_dim, group, self.input_attributes, self.reg, name=group_name) for
            group_name, group in self.feature_group_list]
        self.din_layer = DinTower(self.config, input_to_wide_emb_list)

    def call(self, inputs, training=None, mask=None):
        pred = self.din_layer(inputs)
        return pred
