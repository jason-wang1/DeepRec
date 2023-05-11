import tensorflow as tf
from tensorflow import keras
from train_pipeline import TrainPipeline


def mask_feature(feat):
    def func(feature_dict, label):
        feature_dict[feat]["value"] = tf.zeros_like(feature_dict[feat]["value"])
        return feature_dict, label
    return func


def feature_evaluation():
    date_time = "2023-05-10 180849"
    model_path = f"../output/{date_time}\\model"
    model = keras.models.load_model(model_path)
    result = []

    # data_config_path = "..\config\data_csv_test\data_csv_test.json"
    # model_config_path = "..\config\data_csv_test\model_esmm.json"
    data_config_path = "..\config\data_aliyun_down_rec\data_aliyun_down_rec.json"
    model_config_path = "..\config\data_aliyun_down_rec\model_esmm.json"
    pipeline = TrainPipeline(data_config_path, model_config_path, run_eagerly=False)
    valid_ds = pipeline.read_data(data_type="valid")
    metrics = model.evaluate(valid_ds, return_dict=True)
    result.append(("None", metrics["output_2_auc_1"], 0.0))
    for feature in pipeline.feature_fields:
        print(f"======== mask feature: {feature} ========")
        valid_ds = pipeline.read_data(data_type="valid")
        valid_ds = valid_ds.map(mask_feature(feature))
        metrics = model.evaluate(valid_ds, return_dict=True)
        result.append((feature, metrics["output_2_auc_1"], result[0][1] - metrics["output_2_auc_1"]))
        print(result)
    result.sort(key=lambda x: x[1])
    # 该模型的特征重要度结果，越靠前越重要
    with open(f"..\\output\\{date_time}\\feature_evaluation.csv", 'w', encoding='utf-8') as f:
        for tup in result:
            f.write(f"{tup[0]},{tup[1]},{tup[2]}\n")


if __name__ == '__main__':
    feature_evaluation()
