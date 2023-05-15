from train_pipeline import TrainPipeline


def save_model():
    date_time = pipeline.date_time
    model.save(f"..\\output\\{date_time}\\model")
    with open(f"..\\output\\{date_time}\\data_config.json", 'w', encoding='utf-8') as f:
        f.write(pipeline.data_config_str)
    with open(f"..\\output\\{date_time}\\model_config.json", 'w', encoding='utf-8') as f:
        f.write(pipeline.model_config_str)


if __name__ == '__main__':
    data_config_path = "..\config\data_aliyun_down_rec\data_aliyun_down_rec.json"
    # model_config_path = "..\config\data_aliyun_down_rec\model_esmm.json"
    # model_config_path = "..\config\data_aliyun_down_rec\model_din.json"
    # model_config_path = "..\config\data_aliyun_down_rec\model_din_esmm.json"
    model_config_path = "..\config\data_aliyun_down_rec\model_twotower_deepfm.json"
    # data_config_path = "..\config\data_csv_test\data_csv_test.json"
    # model_config_path = "..\config\data_csv_test\model_afm.json"
    # model_config_path = "..\config\data_csv_test\model_din.json"
    pipeline = TrainPipeline(data_config_path, model_config_path, run_eagerly=False)
    model = pipeline.train()
    model.summary()
    save_model()
