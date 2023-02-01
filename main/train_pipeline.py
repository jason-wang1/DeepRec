import sys

import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from models.twotower_deepfm import TwoTowerDeepFM
from models.deepfm import DeepFM

class TrainPipeline():
    def __init__(self, config):
        self.config = config

    def read_data(self):
        data_config = self.config["data_config"]
        input_fields = data_config["input_fields"]
        column_names = [e["input_name"] for e in input_fields]
        column_defaults = [e["default_val"] for e in input_fields]
        if data_config["input_type"] == "CSVInput":
            train_ds = tf.data.experimental.make_csv_dataset(
                self.config["train_input_path"], data_config["batch_size"], column_names=column_names,
                column_defaults=column_defaults, label_name=data_config["label_fields"])
        else:
            print(f"unexpected input_type: {data_config['input_type']}")
            sys.exit(1)
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
        model_type = self.config["model_type"]
        if model_type == "TwoTowerDeepFM":
            model = TwoTowerDeepFM(self.config)
        elif model_type == "DeepFM":
            model = DeepFM(self.config)
        else:
            print(f"unexpected model_type: {model_type}")
            sys.exit(1)
        model.compile(optimizer=optimizer, loss=self.config["train_config"]["loss"],
                      metrics=self.config["train_config"]["metrics"])
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
