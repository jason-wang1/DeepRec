import sys
import time
import math
from collections import OrderedDict
import json
import tensorflow as tf
from tensorflow.python.keras import optimizers
from models.twotower_deepfm import TwoTowerDeepFM
from models.deepfm import DeepFM
from models.afm import AFM
from models.fm import FM
from models.esmm import ESMM
from models.din import DIN


class TrainPipeline:
    def __init__(self, data_config_path, model_config_path, run_eagerly=None):
        self.date_time = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
        self.config_str = ""
        with open(data_config_path) as f:
            config_str = f.read()
            data_config = json.loads(config_str, object_pairs_hook=OrderedDict)
            self.config_str += config_str + "\n"
        with open(model_config_path) as f:
            config_str = f.read()
            model_config = json.loads(config_str, object_pairs_hook=OrderedDict)
            self.config_str += config_str
        self.config = {**data_config, **model_config}
        self.run_eagerly = run_eagerly
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    def map_sample(self):
        feature_fields = set()
        for feature_groups in self.config["model_config"]["feature_groups"].values():
            for feature in feature_groups:
                if isinstance(feature['input_names'], list):
                    feature_fields.update(feature['input_names'])
                elif isinstance(feature['input_names'], str):
                    feature_fields.add(feature['input_names'])
                else:
                    raise ValueError(f"input_names of feature_fields type error: {feature}")
        input_attributes = self.config["data_config"]["input_attributes"]
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
            for feature in feature_fields:
                tensor = sample_tensor_dict[feature]
                if input_attributes[feature]["field_type"] == "WeightTagFeature":
                    pad_num = input_attributes[feature]["pad_num"]
                    tensor = tf.strings.split(tf.strings.split(tensor, chr(1)), chr(2))
                    tensor = tensor.to_tensor(shape=[batch_size, pad_num, 2], default_value='')
                    index, value = tf.split(tensor, num_or_size_splits=2, axis=2)
                    index = tf.squeeze(index, axis=2)
                    value = tf.squeeze(value, axis=2)
                    emp_str = tf.constant('', shape=[batch_size, pad_num])
                    value_0 = tf.equal(value, emp_str)
                    value = tf.where(value_0, tf.constant('0', shape=[batch_size, pad_num]), value)
                    value = tf.strings.to_number(value, out_type=tf.float32)
                    if input_attributes[feature]["index_type"] == "int":
                        index = tf.where(tf.equal(index, emp_str), tf.constant('0', shape=[batch_size, pad_num]), index)
                        index = tf.strings.to_number(index, out_type=tf.int32)
                        res = {"index": index, "value": value}
                    elif input_attributes[feature]["index_type"] == "string":
                        res = {"index": index, "value": value}
                    else:
                        raise ValueError(f"unexpected feature index_type{feature['index_type']}")
                elif input_attributes[feature]["field_type"] == "TagFeature":
                    pad_num = input_attributes[feature]["pad_num"]
                    index = tf.strings.split(tensor, chr(1))
                    index = index.to_tensor(shape=[batch_size, pad_num], default_value='')
                    emp_str = tf.constant('', shape=[batch_size, pad_num])
                    value_0 = tf.equal(index, emp_str)
                    value = tf.where(value_0, tf.zeros_like(index, dtype=tf.float32),
                                     tf.ones_like(index, dtype=tf.float32))
                    if input_attributes[feature]["index_type"] == "int":
                        index = tf.where(tf.equal(index, emp_str), tf.constant('0', shape=[batch_size, pad_num]), index)
                        index = tf.strings.to_number(index, out_type=tf.int32)
                        res = {"index": index, "value": value}
                    elif input_attributes[feature]["index_type"] == "string":
                        res = {"index": index, "value": value}
                    else:
                        raise ValueError(f"unexpected feature index_type{input_attributes[feature]['index_type']}")
                else:
                    index = tf.reshape(tensor, shape=[batch_size, 1])
                    res = {"index": index, "value": tf.ones_like(index, dtype=tf.float32)}
                feature_dict[feature] = res
            return feature_dict, get_label(sample_tensor_dict)

        return func

    def get_default_value(self, input_attribute):
        if input_attribute["field_type"] == "ClassLabel":
            return 0
        elif input_attribute["field_type"] == "ContinuousLabel":
            return 0.0
        elif input_attribute["field_type"] in ["WeightTagFeature", "TagFeature"]:
            return ""
        elif input_attribute["field_type"] == "SingleFeature":
            if input_attribute["index_type"] == "int":
                return 0
            elif input_attribute["index_type"] == "float":
                return 0.0
            elif input_attribute["index_type"] == "string":
                return ""
            else:
                raise ValueError(f"unexpected field_type: {input_attribute['field_type']}")
        else:
            raise ValueError(f"unexpected field_type: {input_attribute['field_type']}")

    def get_default_value_list(self):
        input_attributes = self.config["data_config"]["input_attributes"]
        column_defaults = []
        for input_field, input_attribute in input_attributes.items():
            column_defaults.append(self.get_default_value(input_attribute))
        return column_defaults

    def read_csv_data(self, data_type):
        input_fields = [input_field for input_field in self.config["data_config"]["input_attributes"]]
        print(f"input_fields: {input_fields}")
        column_defaults = self.get_default_value_list()
        if data_type == "train":
            dataset = tf.data.experimental.CsvDataset(
                self.config["train_input_path"], column_defaults)
        elif data_type == "valid":
            dataset = tf.data.experimental.CsvDataset(
                self.config["valid_input_path"], column_defaults)
        else:
            raise ValueError(f"unexpected data_type: {data_type}")

        def convert_to_dict(*tup_sample):
            res = {}
            for i, input_field in enumerate(input_fields):
                res[input_field] = tup_sample[i]
            return res
        dataset = dataset.map(convert_to_dict)
        return dataset

    def read_aliyun_data(self, data_type):
        input_fields = []
        output_types = {}
        for key, attr_dict in self.config["data_config"]["input_attributes"].items():
            if attr_dict["field_type"].endswith("Feature"):
                aliyun_field = "feat_" + key
            else:
                aliyun_field = key
            input_fields.append((aliyun_field, key))
            if attr_dict["field_type"] == "SingleFeature":
                if attr_dict["index_type"] == "int":
                    output_types[key] = tf.int32
                elif attr_dict["index_type"] == "float":
                    output_types[key] = tf.float32
                elif attr_dict["index_type"] == "string":
                    output_types[key] = tf.string
                else:
                    raise ValueError(f"unexpected index_type: {attr_dict['index_type']}")
            elif attr_dict["field_type"] == "ClassLabel":
                output_types[key] = tf.int32
            elif attr_dict["field_type"] == "ContinuousLabel":
                output_types[key] = tf.float32
            elif attr_dict["field_type"] in ["TagFeature", "WeightTagFeature"]:
                output_types[key] = tf.string
            else:
                raise ValueError(f"unexpected index_type: {attr_dict['index_type']}")
        print(input_fields)
        print(output_types)
        from config.account import access_id, secret_access_key, project, endpoint
        from odps import ODPS
        o = ODPS(access_id, secret_access_key, project, endpoint=endpoint)
        def read_max_compute():
            with o.execute_sql(sql).open_reader() as reader:
                for record in reader:
                    res = {}
                    for aliyun_field, key in input_fields:
                        if record[aliyun_field]:
                            res[key] = record[aliyun_field]
                        else:
                            res[key] = self.get_default_value(self.config["data_config"]["input_attributes"][key])
                    yield res

        if data_type == "train":
            sql = f"""
            SELECT  *
            FROM    {self.config["data_config"]['table_name']}
            WHERE   dt BETWEEN {self.config["data_config"]['train_start_dt']} AND {self.config["data_config"]['train_end_dt']}
            ORDER BY RAND()
            """
            dataset = tf.data.Dataset.from_generator(read_max_compute, output_types=output_types)
        elif data_type == "valid":
            sql = f"""
            SELECT  *
            FROM    {self.config["data_config"]['table_name']}
            WHERE   dt BETWEEN {self.config["data_config"]['valid_start_dt']} AND {self.config["data_config"]['valid_end_dt']}
            LIMIT   {self.config["data_config"]['valid_limit']}
            """
            self.valid_data = [sample for sample in read_max_compute()]
            def valid_data_gen():
                for sample in self.valid_data:
                    yield sample
            dataset = tf.data.Dataset.from_generator(valid_data_gen, output_types=output_types)
        else:
            raise ValueError(f"unexpected data_type: {data_type}")

        return dataset

    def read_data(self, data_type):
        data_config = self.config["data_config"]
        if data_config["input_type"] == "CSVInput":
            train_ds = self.read_csv_data(data_type)
        elif data_config["input_type"] == "MaxComputeInput":
            train_ds = self.read_aliyun_data(data_type)
        else:
            raise ValueError(f"unexpected input_type: {data_config['input_type']}")
        if data_type == "train":
            train_ds = train_ds.shuffle(10000, reshuffle_each_iteration=True)
            train_ds = train_ds.repeat(self.config["data_config"].get("repeat", 1))
        train_ds = train_ds.batch(batch_size=self.config["data_config"]["batch_size"], drop_remainder=True)
        train_ds = train_ds.map(self.map_sample(), num_parallel_calls=4)
        train_ds = train_ds.prefetch(data_config["prefetch_size"])
        return train_ds

    def get_optimizer(self):
        optimizer_config = self.config["train_config"]["optimizer"]
        learning_rate = optimizer_config["learning_rate"]
        if isinstance(learning_rate, float):
            lr_schedule = learning_rate
        else:
            raise ValueError(f"unexpected config type: {learning_rate}")
        if optimizer_config["name"] == "adam":
            optimizer = optimizers.adam_v2.Adam(learning_rate=lr_schedule)
        else:
            raise ValueError(f"unexpected optimizer name: {optimizer_config['name']}")
        return optimizer

    def get_callbacks(self):
        patience = self.config["train_config"].get("early_stopping_patience", math.ceil(self.config["train_config"]["epochs"]/10))
        log_dir = f"..\\output\\{self.date_time}\\tensorboard"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            # update_freq=5000,
            histogram_freq=1
        )
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            patience=patience,
            restore_best_weights=True
        )
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath="..\output\checkpoint_model",
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        return [tensorboard_callback, early_stopping_callback, checkpoint_cb]

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
        train_ds = self.read_data(data_type="train")
        valid_ds = self.read_data(data_type="valid")
        model = self.get_model()
        model.fit(
            x=train_ds,
            epochs=self.config["train_config"]["epochs"],
            callbacks=self.get_callbacks(),
            steps_per_epoch=self.config["train_config"]["steps_per_epoch"],
            validation_data=valid_ds
        )
        return model


@tf.function
def get_one_sample():
    ds = pipeline.read_data(data_type="train")
    for sample in ds:
        for feature_name, feature_tensor in sample[0].items():
            print(f"{feature_name}: {feature_tensor}")
        print(sample[1])
        print(sample)
        break


if __name__ == '__main__':
    # 预览训练样本
    # pipeline = TrainPipeline("..\config\data_aliyun_down_rec\data_aliyun_down_rec.json",
    #                          "..\config\data_aliyun_down_rec\model_esmm.json", run_eagerly=False)
    pipeline = TrainPipeline("..\config\data_csv_test\data_csv_test.json",
                             "..\config\data_csv_test\model_esmm.json", run_eagerly=False)
    print(pipeline.config_str)
    get_one_sample()
