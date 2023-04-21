import sys
import json
import time
import tensorflow as tf
from tensorflow.python.keras import models
from collections import OrderedDict

if __name__ == '__main__':
    model_path = "../output/2023-04-19 140917\\model"
    model = models.load_model(model_path)
    inputs = {}
    inputs['101'] = tf.constant([356055, 421878, 358992])
    inputs['109_14'] = {
        "index": tf.constant([
            [450954, 450877, 449079, 446271, 450276],
            [446296, 450654, 453931, 453930, 445398],
            [451616, 456727, 451532, 451534, 450462]]),
        "value": tf.constant([
            [0.69315, 1.09861, 2.30259, 1.60944, 1.09861],
            [1.09861, 0.69315, 2.48491, 1.09861, 0.69315],
            [4.67283, 0.69315, 1.09861, 0.69315, 0.69315]
        ])}
    inputs['210'] = tf.ragged.constant([
        [9093481, 9103049, 9116612, 9107458, 9078913, 9066813, 9060055],
        [9019411, 9063881, 9040145, 9089922, 9033260, 9111124, 9080388, 9050902, 9052837, 9020727],
        [9073568, 9101513, 9020340, 9096388, 9066941, 9064040, 9078619]
    ])
    print(inputs)
    outputs = model(inputs)
    print(outputs)










    req = {"instances": [{"input_index": [0, 0, 0, 0, 0, 0, 0, 0], "input_value": [0.0, 0.0, 0.0, 0.0]}],
           "signature_name": "serving_default"}
    req = {"instances": [
        {"101": [356055], "109_14": {"index": [450954, 450877, 449079, 446271, 450276], "value": [0.69315, 1.09861, 2.30259, 1.60944, 1.09861]}, "210": [9093481, 9103049, 9116612, 9107458, 9078913, 9066813, 9060055]}
    ],"signature_name": "serving_default"}
