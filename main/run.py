import json
from train_pipeline import TrainPipeline

two_tower_deepfm = "deepfm_twotower_on_taobao.json"
deep_fm = "deepfm_on_avazu.json"
json_path = f"..\\config\\{deep_fm}"
with open(json_path) as f:
    config_str = f.read()
    config = json.loads(config_str)

pipeline = TrainPipeline(config)
model = pipeline.train()
model.summary()
