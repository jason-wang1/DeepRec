import sys
# import datetime
sys.path.append("C:\\Users\\BoWANG\\PycharmProjects\\DeepRec")
# sys.path.append("/home/wangbo/DeepRec/protos")

import json
import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from models.deepfm import DeepFM


def print_ds(dataset):
    i = 0
    for item in dataset:
        print(item)
        print(item[0]["hour"])
        # print(item[0]["site_domain"])
        # print(item[0]["site_domain"].shape)
        i += 1
        if i == 1:
            break
def get_hour_feat(data_dict, label):
    hour_tensor = data_dict["hour"]
    hour_tensor = hour_tensor % 100
    data_dict["hour"] = hour_tensor
    return data_dict, label

json_path = "deepfm_on_avazu.json"
with open(json_path) as f:
    config_str = f.read()
    config = json.loads(config_str)

train_ds = tf.data.experimental.make_csv_dataset(
    config["train_input_path"], batch_size=config["data_config"]["batch_size"], label_name=config["data_config"]["label_fields"])
train_ds = train_ds.map(get_hour_feat)
# print_ds(train_ds)
# for feature_batch, label_batch in train_ds.take(1):
#   print("'survived': {}".format(label_batch))
#   print("features:")
#   for key, value in feature_batch.items():
#     print("  {!r:20s}: {}".format(key, value))

model = DeepFM(config)
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
