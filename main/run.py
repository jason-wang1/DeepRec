import sys
import json
from train_pipeline import TrainPipeline


data_config_path = sys.argv[1]
with open(data_config_path) as f:
    config_str = f.read()
    data_config = json.loads(config_str)

model_config_path = sys.argv[2]
with open(model_config_path) as f:
    config_str = f.read()
    model_config = json.loads(config_str)

config = {**data_config, **model_config}
print(config)
pipeline = TrainPipeline(config, run_eagerly=False)
model = pipeline.train()
model.summary()
