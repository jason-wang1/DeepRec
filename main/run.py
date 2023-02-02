import sys
import json
from train_pipeline import TrainPipeline


config_path = sys.argv[1]
with open(config_path) as f:
    config_str = f.read()
    config = json.loads(config_str)

print(config)
pipeline = TrainPipeline(config, run_eagerly=False)
model = pipeline.train()
model.summary()
