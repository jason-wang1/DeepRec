from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l1, l2


class Base(Model):
    def __init__(self, config, **kwargs):
        self.config = config
        self.input_attributes = config["data_config"]["input_attributes"]
        self.emb_dim = config["model_config"]["embedding_dim"]
        if "l2_reg" in config["model_config"]:
            self.reg = l2(config["model_config"]["l2_reg"])
        elif "l1_reg" in config["model_config"]:
            self.reg = l1(config["model_config"]["l1_reg"])
        else:
            self.reg = None
        self.feature_group_list = [(group_name + "_input", group) for group_name, group in config["model_config"]["feature_groups"].items() if group_name != "din"]
        super(Base, self).__init__(**kwargs)


class BaseLayer(Layer):
    def __init__(self, config, **kwargs):
        self.config = config
        self.input_attributes = config["data_config"]["input_attributes"]
        self.emb_dim = config["model_config"]["embedding_dim"]
        if "l2_reg" in config["model_config"]:
            self.reg = l2(config["model_config"]["l2_reg"])
        elif "l1_reg" in config["model_config"]:
            self.reg = l1(config["model_config"]["l1_reg"])
        else:
            self.reg = None
        self.feature_group_list = [(group_name + "_input", group) for group_name, group in config["model_config"]["feature_groups"].items() if group_name != "din"]
        super(BaseLayer, self).__init__(**kwargs)
