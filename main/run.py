import time
from train_pipeline import TrainPipeline


def save_model():
    date_time = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
    model.save(f"..\\output\\{date_time}\\model")
    with open(f"..\\output\\{date_time}\\config.json", 'w', encoding='utf-8') as f:
        f.write(pipeline.config_str)


if __name__ == '__main__':
    # data_config_path = "..\config\data_aliyun_down_rec\data_aliyun_down_rec.json"
    # model_config_path = "..\config\data_aliyun_down_rec\model_esmm.json"
    data_config_path = "..\config\data_csv_test\data_csv_test.json"
    model_config_path = "..\config\data_csv_test\model_esmm.json"
    pipeline = TrainPipeline(data_config_path, model_config_path, run_eagerly=True)
    model = pipeline.train()
    model.summary()
    # save_model()
