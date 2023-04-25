import sys
import json
import time
from train_pipeline import TrainPipeline


def get_config():
    data_config_path = sys.argv[1]
    with open(data_config_path) as f:
        config_str = f.read()
        data_config = json.loads(config_str)

    model_config_path = sys.argv[2]
    with open(model_config_path) as f:
        config_str = f.read()
        model_config = json.loads(config_str)

    config = {**data_config, **model_config}
    return config


def train():
    model = pipeline.train()
    model.summary()
    date_time = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
    model.save(f"..\\output\\{date_time}\\model")
    with open(f"..\\output\\{date_time}\\config.json", 'w', encoding='utf-8') as f:
        f.write(res_config)


def print_train_data():
    ds = pipeline.read_data()
    sample = next(iter(ds))
    for k, v in sample[0].items():
        print(f"{k}: {v}")
    print(sample[1])


if __name__ == '__main__':
    config = get_config()
    res_config = json.dumps(config)  # 序列化成json格式字符串
    pipeline = TrainPipeline(config, run_eagerly=True)
    print_train_data()
    # train()
    print(config)
