import sys
from collections import OrderedDict
import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from models.twotower_deepfm import TwoTowerDeepFM
from models.deepfm import DeepFM
from models.afm import AFM
from models.fm import FM
from models.esmm import ESMM
from models.din import DIN

class TrainPipeline():
    def __init__(self, config, run_eagerly=None):
        self.config = config
        self.run_eagerly = run_eagerly

    def map_weight_tag_feature(self, weight_tag_list, pad_num):
        def func(tensor_dict):
            for weight_tag in weight_tag_list:
                tensor = tensor_dict[weight_tag]
                tensor = tf.strings.split(tf.strings.split(tensor, "|"), ":")
                tensor = tensor.to_tensor(shape=[None, pad_num, 2], default_value='0')
                index, value = tf.split(tensor, num_or_size_splits=2, axis=2)
                res = {"index": tf.squeeze(tf.strings.to_number(index, out_type=tf.int32), axis=2),
                       "value": tf.squeeze(tf.strings.to_number(value, out_type=tf.float32), axis=2)}
                tensor_dict[weight_tag] = res
            return tensor_dict
        return func

    def map_tag_feature(self, tag_list):
        def func(tensor_dict):
            for weight_tag in tag_list:
                tensor = tensor_dict[weight_tag]
                tensor = tf.strings.to_number(tf.strings.split(tensor, "|"), out_type=tf.int32)
                tensor_dict[weight_tag] = tensor
            return tensor_dict
        return func

    def get_label(self, labels):
        def get_label(tensor_dict):
            label = tensor_dict[labels]
            del tensor_dict[labels]
            return tensor_dict, label
        def get_multi_label(tensor_dict):
            label_dict = {}
            for label_name in labels:
                label_dict[label_name] = tensor_dict[label_name]
                del tensor_dict[label_name]
            return tensor_dict, label_dict
        if isinstance(labels, str):
            return get_label
        elif isinstance(labels, list):
            if len(labels) == 1:
                labels = labels[0]
                return get_label
            else:
                return get_multi_label
        else:
            raise ValueError(f"labels must be a string or list, but {labels}")

    def read_data(self):
        data_config = self.config["data_config"]
        input_fields = data_config["input_fields"]
        column_names = [e["input_name"] for e in input_fields]
        column_defaults = [e["default_val"] for e in input_fields]
        if data_config["input_type"] == "CSVInput":
            train_ds = tf.data.experimental.make_csv_dataset(
                self.config["train_input_path"], data_config["batch_size"], column_names=column_names,
                shuffle_buffer_size=100000, column_defaults=column_defaults)
        else:
            print(f"unexpected input_type: {data_config['input_type']}")
            sys.exit(1)
        feature_label_list = [feature["input_names"] for feature in self.config["feature_config"]["features"]]
        feature_label_list.extend(self.config["data_config"]["label_fields"])
        train_ds = train_ds.map(self.map_feature(feature_label_list), num_parallel_calls=4)
        weight_tag_list = [e["input_names"] for e in self.config["feature_config"]["features"] if e["feature_type"] == "WeightTagFeature"]
        if weight_tag_list:
            pad_num = self.config["feature_config"]["pad_num"]
            train_ds = train_ds.map(self.map_weight_tag_feature(weight_tag_list, pad_num), num_parallel_calls=4)
        tag_list = [e["input_names"] for e in self.config["feature_config"]["features"] if e["feature_type"] == "TagFeature"]
        if tag_list:
            train_ds = train_ds.map(self.map_tag_feature(tag_list), num_parallel_calls=4)
        train_ds = train_ds.map(self.get_label(data_config["label_fields"]), num_parallel_calls=4)
        train_ds = train_ds.prefetch(data_config["prefetch_size"])
        return train_ds

    def get_optimizer(self):
        optimizer_config = self.config["train_config"]["optimizer"]
        learning_rate = optimizer_config["learning_rate"]
        if isinstance(learning_rate, dict):
            if learning_rate["schedule"] == "exponential_decay":
                lr_schedule = ExponentialDecay(
                    learning_rate["initial_learning_rate"], learning_rate["decay_steps"], learning_rate["decay_rate"]
                )
            else:
                print(f"unexpected schedule: {learning_rate['schedule']}")
                sys.exit(1)
        elif isinstance(learning_rate, float):
            lr_schedule = learning_rate
        else:
            print(f"unexpected config type: {learning_rate}")
            sys.exit(1)
        if optimizer_config["name"] == "adam":
            optimizer = optimizers.adam_v2.Adam(learning_rate=lr_schedule)
        else:
            print(f"unexpected optimizer name: {optimizer_config['name']}")
            sys.exit(1)
        return optimizer

    def get_model(self):
        optimizer = self.get_optimizer()
        model_type = self.config["model_config"]["model_type"]
        if model_type == "FM":
            model = FM(self.config)
        elif model_type == "TwoTowerDeepFM":
            model = TwoTowerDeepFM(self.config)
        elif model_type == "DeepFM":
            model = DeepFM(self.config)
        elif model_type == "AFM":
            model = AFM(self.config)
        elif model_type == "ESMM":
            model = ESMM(self.config)
        elif model_type == "DIN":
            model = DIN(self.config)
        else:
            print(f"unexpected model_type: {model_type}")
            sys.exit(1)
        if model_type in ["ESMM"]:
            model.compile(optimizer=optimizer, run_eagerly=self.run_eagerly)
        else:
            model.compile(optimizer=optimizer, loss=self.config["train_config"]["loss"],
                          metrics=self.config["train_config"]["metrics"], run_eagerly=self.run_eagerly)
        return model

    def train(self):
        train_ds = self.read_data()
        model = self.get_model()
        model.fit(
            x=train_ds,
            epochs=self.config["train_config"]["epochs"],
            steps_per_epoch=self.config["train_config"]["steps_per_epoch"],
        )
        return model

    def map_feature(self, features):
        def func(tensor_dict):
            res = {}
            for feature in features:
                res[feature] = tensor_dict[feature]
            return res
        return func
