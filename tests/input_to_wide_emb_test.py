import tensorflow as tf
import numpy as np
from collections import OrderedDict
from layers.input_to_wide_emb import WeightTagPoolingInput, TagPoolingInput, IdInput, RawInput, ComboInput

class InputToWideEmbTest(tf.test.TestCase):
    def __init__(self, methodName='InputToWideEmbTest'):
        super(InputToWideEmbTest, self).__init__(methodName=methodName)

    def test_weight_tag_pooling_input(self):
        inputs = OrderedDict()
        inputs["index"] = tf.constant([[2702747,  826126, 1081350,  630553, 2996873],
                                       [3038243,  962914, 1523483,  975736, 3425351],
                                       [ 791878, 2739489,  732942, 1469720, 2948413]])
        inputs["value"] = tf.constant([[0.69315, 1.09861, 0.69315, 2.19722, 0.69315],
                                       [1.60944, 0.69315, 1.09861, 0.69315, 1.94591],
                                       [0.69315, 0.69315, 0.69315, 0.69315, 0.69315]])
        feat = {"input_names": "110_14", "feature_type": "WeightTagFeature", "hash_bucket_size": 100000}
        emb_dim = 16
        layer = WeightTagPoolingInput(feat, emb_dim, None, True)
        wide, deep = layer(inputs)
        print(wide)
        print(deep)
        deep_shape = np.ones([3, emb_dim])
        wide_shape = np.ones([3])
        self.assertShapeEqual(wide_shape, wide)
        self.assertShapeEqual(deep_shape, deep)

    def test_tag_pooling_input(self):
        inputs = tf.ragged.constant([[9077964, 9088724, 9063310, 9078970, 9113308, 9072542, 9052341, 9038775],
                                     [9021735, 9064965, 9115623, 9073354],
                                     [9042585, 9088568, 9097943, 9044661, 9078989, 9053181, 9097981]])
        feat = {"input_names": "210", "feature_type": "TagFeature", "hash_bucket_size": 100000}
        emb_dim = 16
        layer = TagPoolingInput(feat, emb_dim, None, True)
        wide, deep = layer(inputs)
        print(wide)
        print(deep)
        wide_shape = np.ones([3])
        deep_shape = np.ones([3, emb_dim])
        self.assertShapeEqual(wide_shape, wide)
        self.assertShapeEqual(deep_shape, deep)

        inputs = tf.constant([[9077964, 9088724, 9063310, 9078970],
                              [9021735, 9064965, 9115623, 9073354],
                              [9042585, 9088568, 9097943, 9044661]])
        wide, deep = layer(inputs)
        self.assertShapeEqual(wide_shape, wide)
        self.assertShapeEqual(deep_shape, deep)

    def test_id_input(self):
        inputs = tf.constant([7941622, 4558092, 4308727])
        feat = {"input_names": "129", "feature_type": "IdFeature", "hash_bucket_size": 10}
        emb_dim = 16
        layer = IdInput(feat, emb_dim, None, True)
        wide, deep = layer(inputs)
        print(wide)
        print(deep)
        wide_shape = np.ones([3])
        deep_shape = np.ones([3, emb_dim])
        self.assertShapeEqual(wide_shape, wide)
        self.assertShapeEqual(deep_shape, deep)

    def test_raw_input(self):
        inputs = tf.constant([290., 291., 23.61])
        feat = {"input_names": "price", "feature_type": "RawFeature", "boundaries": [49.0, 139.0, 352.0]}
        emb_dim = 16
        layer = RawInput(feat, emb_dim, None, True)
        wide, deep = layer(inputs)
        print(wide)
        print(deep)
        wide_shape = np.ones([3])
        deep_shape = np.ones([3, emb_dim])
        self.assertShapeEqual(wide_shape, wide)
        self.assertShapeEqual(deep_shape, deep)

    def test_comb_input(self):
        inputs = [tf.constant([b'x', b'y', b'z']),
                  tf.constant([b'1', b'2', b'3'])]
        print(inputs)
        feat = {"input_names": ["site_id", "app_id"], "feature_name": "site_id_app_id", "feature_type": "ComboFeature", "hash_bucket_size": 1000}
        emb_dim = 16
        layer = ComboInput(feat, emb_dim, None, True)
        wide, deep = layer(inputs)
        print(wide)
        print(deep)
        wide_shape = np.ones([3])
        deep_shape = np.ones([3, emb_dim])
        self.assertShapeEqual(wide_shape, wide)
        self.assertShapeEqual(deep_shape, deep)


if __name__ == '__main__':
    tf.test.main()
