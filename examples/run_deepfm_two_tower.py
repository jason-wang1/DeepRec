import sys
# import datetime
sys.path.append("C:\\Users\\BoWANG\\PycharmProjects\\DeepRec")
# sys.path.append("/home/wangbo/DeepRec/protos")

import json
import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from models.twotower_deepfm import TwoTowerDeepFM


json_path = "..\\config\\deepfm_twotower_on_taobao.json"
with open(json_path) as f:
    config_str = f.read()
    config = json.loads(config_str)

input_fields = config["data_config"]["input_fields"]
column_names = [e["input_name"] for e in input_fields]
column_defaults = [e["default_val"] for e in input_fields]
train_ds = tf.data.experimental.make_csv_dataset(
    config["train_input_path"], config["data_config"]["batch_size"],  column_names=column_names,
    column_defaults=column_defaults, label_name=config["data_config"]["label_fields"])

train_ds = train_ds.prefetch(config["data_config"]["prefetch_size"])
model = TwoTowerDeepFM(config)
learning_rate = config["train_config"]["optimizer"]["learning_rate"]
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
optimizer = optimizers.adam_v2.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss=config["train_config"]["loss"], metrics=config["train_config"]["metrics"])

model.fit(
    x=train_ds,
    epochs=config["train_config"]["epochs"],
    steps_per_epoch=config["train_config"]["steps_per_epoch"],
)
model.summary()
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# model.save(config["model_dir"] + current_time)
