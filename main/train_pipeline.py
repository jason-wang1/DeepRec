import sys
import json
import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from models.twotower_deepfm import TwoTowerDeepFM
from models.deepfm import DeepFM
from models.afm import AFM
from models.fm import FM
from models.esmm import ESMM
from models.din import DIN


class TrainPipeline:
    def __init__(self, data_config_path, model_config_path, run_eagerly=None):
        with open(data_config_path) as f:
            config_str = f.read()
            data_config = json.loads(config_str)
        with open(model_config_path) as f:
            config_str = f.read()
            model_config = json.loads(config_str)
        self.config = {**data_config, **model_config}
        self.config_str = json.dumps(self.config)
        self.run_eagerly = run_eagerly

    def map_sample(self):
        features = self.config["feature_config"]["features"]
        pad_num = self.config["feature_config"]["pad_num"]
        batch_size = self.config["data_config"]["batch_size"]

        def get_label(tensor_dict):
            labels = self.config["data_config"]["label_fields"]
            if isinstance(labels, str):
                return tensor_dict[labels]
            elif isinstance(labels, list):
                if len(labels) == 1:
                    labels = labels[0]
                    return tensor_dict[labels]
                else:
                    label_dict = {}
                    for label_name in labels:
                        label_dict[label_name] = tensor_dict[label_name]
                    return label_dict
            else:
                raise ValueError(f"labels must be a string or list, but {labels}")

        def func(sample_tensor_dict):
            feature_dict = {}
            for feature in features:
                if feature["feature_type"] == "WeightTagFeature":
                    tensor = sample_tensor_dict[feature["input_names"]]
                    tensor = tf.strings.split(tf.strings.split(tensor, chr(1)), chr(2))
                    tensor = tensor.to_tensor(shape=[None, pad_num, 2], default_value='')
                    index, value = tf.split(tensor, num_or_size_splits=2, axis=2)
                    index = tf.squeeze(index, axis=2)
                    value = tf.squeeze(value, axis=2)
                    emp_str = tf.constant('', shape=[batch_size, pad_num])
                    value_0 = tf.equal(value, emp_str)
                    value = tf.where(value_0, tf.constant('0', shape=[batch_size, pad_num]), value)
                    value = tf.strings.to_number(value, out_type=tf.float32)
                    if feature["index_type"] == "int":
                        index = tf.where(tf.equal(index, emp_str), tf.constant('0', shape=[batch_size, pad_num]), index)
                        index = tf.strings.to_number(index, out_type=tf.int32)
                        res = {"index": index, "value": value}
                    elif feature["index_type"] == "string":
                        res = {"index": index, "value": value}
                    else:
                        raise ValueError(f"unexpected feature index_type{feature['index_type']}")
                elif feature["feature_type"] == "TagFeature":
                    index = sample_tensor_dict[feature["input_names"]]
                    index = tf.strings.split(index, chr(1))
                    index = index.to_tensor(shape=[None, pad_num], default_value='')
                    emp_str = tf.constant('', shape=[batch_size, pad_num])
                    value_0 = tf.equal(index, emp_str)
                    value = tf.where(value_0, tf.zeros_like(index, dtype=tf.float32),
                                     tf.ones_like(index, dtype=tf.float32))
                    if feature["index_type"] == "int":
                        index = tf.where(tf.equal(index, emp_str), tf.constant('0', shape=[batch_size, pad_num]), index)
                        index = tf.strings.to_number(index, out_type=tf.int32)
                        res = {"index": index, "value": value}
                    elif feature["index_type"] == "string":
                        res = {"index": index, "value": value}
                    else:
                        raise ValueError(f"unexpected feature index_type{feature['index_type']}")
                else:
                    index = sample_tensor_dict[feature["input_names"]]
                    index = tf.expand_dims(index, axis=1)
                    res = {"index": index, "value": tf.ones_like(index, dtype=tf.float32)}
                feature_dict[feature["input_names"]] = res
            return feature_dict, get_label(sample_tensor_dict)

        return func

    def read_data(self):
        data_config = self.config["data_config"]
        input_fields = data_config["input_fields"]
        column_names = [e["input_name"] for e in input_fields]
        column_defaults = [e["default_val"] for e in input_fields]
        if data_config["input_type"] == "CSVInput":
            train_ds = tf.data.experimental.make_csv_dataset(
                self.config["train_input_path"], data_config["batch_size"], column_names=column_names,
                shuffle=False, column_defaults=column_defaults)
        else:
            print(f"unexpected input_type: {data_config['input_type']}")
            sys.exit(1)
        train_ds = train_ds.map(self.map_sample(), num_parallel_calls=4)
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


if __name__ == '__main__':
    # 预览训练样本
    pipeline = TrainPipeline("..\config\data_ali_ccp_test_v1.json", "..\config\model_esmm.json", run_eagerly=True)
    ds = pipeline.read_data()
    sample = next(iter(ds))
    for feature_name, feature_tensor in sample[0].items():
        print(f"{feature_name}: {feature_tensor}")
    print(sample[1])
    print(sample)
