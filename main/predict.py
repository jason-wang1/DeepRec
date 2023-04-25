import sys
import json
import time
import tensorflow as tf
from tensorflow.python.keras import models
from collections import OrderedDict

if __name__ == '__main__':
    model_path = "../output/2023-04-25 100155\\model"
    model = models.load_model(model_path)
    inputs = {}
    inputs['101'] = {
        "index": tf.constant([[356055], [421878], [358992]]),
        "value": tf.constant([[1.], [1.], [1.]])
    }
    inputs['901'] = {
        "index": tf.constant([[0.80049], [0.], [0.12561]]),
        "value": tf.constant([[1.], [1.], [1.]])
    }
    inputs['109_14'] = {
        "index": tf.constant([
            [450954, 450877, 449079, 446271, 0],
            [446296, 450654, 453931, 453930, 445398],
            [451616, 456727, 451532, 451534, 450462]]),
        "value": tf.constant([
            [0.69315, 1.09861, 2.30259, 1.60944, 0.],
            [1.09861, 0.69315, 2.48491, 1.09861, 0.69315],
            [4.67283, 0.69315, 1.09861, 0.69315, 0.69315]
        ])}
    inputs['210'] = {
        "index": tf.constant([
            [9093481, 9103049, 9116612, 9107458, 9078913],
            [9019411, 9063881, 9040145, 9089922, 9033260],
            [0, 0, 0, 0, 0]
        ]),
        "value": tf.constant([
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0.]
        ])
    }
    inputs['902'] = {
        "index": tf.constant([["酒吧台"], ["幼儿园"], ["飞车"]]),
        "value": tf.constant([[1.], [1.], [1.]])
    }
    inputs['903'] = {
        "index": tf.constant([
            ["", "", "", "", ""],
            ["玄关", "水槽", "飞车", "茶室", "格栅"],
            ["流线型酒店", "", "", "", ""]
        ]),
        "value": tf.constant([
            [0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1.],
            [1., 0., 0., 0., 0.]
        ])
    }
    inputs['904'] = {
        "index": tf.constant([
            ["流线型酒店", "电竞房", "欧式拉手", "loft", "旋转楼梯"],
            ["玄关", "水槽", "飞车", "茶室", "格栅"],
            ["流线型酒店", "", "", "", ""]
        ]),
        "value": tf.constant([
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 0., 0., 0., 0.]
        ])
    }
    print(inputs)
    outputs = model(inputs)
    print(outputs)

    req = {"instances": [{"input_index": [0, 0, 0, 0, 0, 0, 0, 0], "input_value": [0.0, 0.0, 0.0, 0.0]}],
           "signature_name": "serving_default"}
    req = {"instances": [
        {"101": [356055], "109_14": {"index": [450954, 450877, 449079, 446271, 450276],
                                     "value": [0.69315, 1.09861, 2.30259, 1.60944, 1.09861]},
         "210": [9093481, 9103049, 9116612, 9107458, 9078913, 9066813, 9060055]}
    ], "signature_name": "serving_default"}
