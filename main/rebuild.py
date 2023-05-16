import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.regularizers import l1, l2

from layers.input_to_wide_emb import InputToWideEmb
from layers.tower_deepfm import TowerDeepFM
from pipeline import TrainPipeline


def get_inputs(feature_groups):
    feature_fields = set()
    for user_feature in feature_groups:
        if isinstance(user_feature['input_names'], list):
            feature_fields.update(user_feature['input_names'])
        elif isinstance(user_feature['input_names'], str):
            feature_fields.add(user_feature['input_names'])
        else:
            raise ValueError(f"input_names of feature_fields type error: {user_feature}")
    inputs_dict = {}
    for input_name in feature_fields:
        pad_num = input_attributes[input_name].get("pad_num", 1)
        default_value = pipeline.get_default_value(input_attributes[input_name])
        if isinstance(default_value, int):
            dtype = tf.int32
        elif isinstance(default_value, float):
            dtype = tf.float32
        elif isinstance(default_value, str):
            dtype = tf.string
        else:
            raise ValueError(f"unexpected default_value: {default_value}")
        inputs_dict[input_name] = {"index": Input(shape=(pad_num,), dtype=dtype),
                                   "value": Input(shape=(pad_num,), dtype=tf.float32)}
    return inputs_dict


def rebuild_deepfm_tower(group, feature_groups):
    input_to_wide_emb = InputToWideEmb(True, emb_dim, feature_groups, input_attributes, reg, name=f"{group}_input")
    dnn_shape = config["model_config"]["deep_hidden_units"]
    tower = TowerDeepFM(group, dnn_shape, reg, name=f"{group}_tower")

    inputs = get_inputs(feature_groups)
    print(inputs)
    wide_input, fm_input = input_to_wide_emb(inputs)
    represent = tower({"wide_input": wide_input, "fm_input": fm_input})

    model = Model(inputs=inputs, outputs=represent)

    ori_model_var_dict = {}
    for var in ori_model.trainable_variables:
        ori_model_var_dict[var.name] = var

    for var in model.trainable_variables:
        org_var_name = 'two_tower_deep_fm/' + var.name
        var.assign(ori_model_var_dict[org_var_name])
    model.save(f"../output/{date_time}\\{group}_model")

    return model


def map_features(feature_groups, feature):
    feature_fields = set()
    for user_feature in feature_groups:
        if isinstance(user_feature['input_names'], list):
            feature_fields.update(user_feature['input_names'])
        elif isinstance(user_feature['input_names'], str):
            feature_fields.add(user_feature['input_names'])
        else:
            raise ValueError(f"input_names of feature_fields type error: {user_feature}")
    feature_dict = {}
    for input_name in feature_fields:
        feature_dict[input_name] = feature[input_name]
    return feature_dict


def check_result():
    dataset = pipeline.read_data("valid")
    for sample in dataset:
        features = sample[0]
        user_feature = map_features(user_feature_groups, features)
        item_feature = map_features(item_feature_groups, features)
        user_represent = user_model(user_feature)
        item_represent = item_model(item_feature)
        pred = tf.reduce_sum(user_represent * item_represent, axis=1, keepdims=True)
        pred = tf.sigmoid(pred)
        ori_pred = ori_model(features)
        tf.assert_equal(pred, ori_pred)
        break


if __name__ == '__main__':
    date_time = "2023-05-16 112832"
    model_path = f"../output/{date_time}\\model"
    ori_model = keras.models.load_model(model_path)
    var_names = [var.name for var in ori_model.trainable_variables]
    print("ori_model:", var_names)

    data_config_path = f"../output/{date_time}/data_config.json"
    model_config_path = f"../output/{date_time}/model_config.json"
    pipeline = TrainPipeline(data_config_path, model_config_path, run_eagerly=False)
    pipeline.config["data_config"]['valid_limit'] = 1024
    pipeline.config["data_config"]["batch_size"] = 512
    config = pipeline.config

    input_attributes = config["data_config"]["input_attributes"]
    emb_dim = config["model_config"]["embedding_dim"]
    if "l2_reg" in config["model_config"]:
        reg = l2(config["model_config"]["l2_reg"])
    elif "l1_reg" in config["model_config"]:
        reg = l1(config["model_config"]["l1_reg"])
    else:
        reg = None

    user_feature_groups = config["model_config"]["feature_groups"]["user"]
    item_feature_groups = config["model_config"]["feature_groups"]["item"]
    # user_model = rebuild_deepfm_tower("user", user_feature_groups)
    # item_model = rebuild_deepfm_tower("item", item_feature_groups)
    user_model = keras.models.load_model(f"../output/{date_time}\\user_model")
    item_model = keras.models.load_model(f"../output/{date_time}\\item_model")

    check_result()
