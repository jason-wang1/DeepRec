import tensorflow as tf
from tensorflow.keras.layers import Embedding
import numpy as np
from collections import OrderedDict
from layers.input_to_wide_emb import AttentionSequencePoolingInput, WeightTagPoolingInput, ComboInput, BaseInputLayer


class InputToWideEmbTest(tf.test.TestCase):
    def __init__(self, methodName='InputToWideEmbTest'):
        super(InputToWideEmbTest, self).__init__(methodName=methodName)

    def test_base_input_num_buckets(self):
        feat = {"input_names": "101", "feature_type": "SingleFeature", "num_buckets": 9999}
        inputs = tf.constant([[2747, 6126, 11350, 1553, 6873],
                              [8243, 2914, 13483, 5736, 5351],
                              [878, 29489, 7942, -9720, 8413]])
        emb_dim = 16
        layer = BaseInputLayer(feat, emb_dim, None, True)
        wide, deep = layer(inputs)
        print(wide)
        print(deep)
        wide_shape = np.ones([3, 5])
        deep_shape = np.ones([3, 5, emb_dim])
        self.assertShapeEqual(wide_shape, wide)
        self.assertShapeEqual(deep_shape, deep)


    def test_weight_tag_pooling_input(self):
        inputs = OrderedDict()
        inputs["index"] = tf.constant([[2702747, 826126, 1081350, 630553, 2996873],
                                       [3038243, 962914, 1523483, 975736, 3425351],
                                       [791878, 2739489, 732942, 1469720, 2948413]])
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

        inputs = OrderedDict()
        inputs["index"] = tf.ragged.constant([[2702747, 826126, 1081350, 630553, 2996873],
                                              [3038243, 962914],
                                              [791878, 2739489, 732942, 1469720]])
        inputs["value"] = tf.ragged.constant([[0.69315, 1.09861, 0.69315, 2.19722, 0.69315],
                                              [1.60944, 0.69315],
                                              [0.69315, 0.69315, 0.69315, 0.69315]])
        wide, deep = layer(inputs)
        print(wide)
        print(deep)
        deep_shape = np.ones([3, emb_dim])
        wide_shape = np.ones([3])
        self.assertShapeEqual(wide_shape, wide)
        self.assertShapeEqual(deep_shape, deep)

    def test_comb_input(self):
        inputs = [{"index": tf.constant([[b'x'], [b'y'], [b'z']]), "value": tf.constant([[1.0], [1.0], [1.0]])},
                  {"index": tf.constant([[b'1'], [b'2'], [b'3']]), "value": tf.constant([[1.0], [1.0], [1.0]])}]
        print(inputs)
        feat = {"input_names": ["site_id", "app_id"], "feature_name": "site_id_app_id_cross",
                "feature_type": "ComboFeature", "hash_bucket_size": 1000}
        emb_dim = 16
        layer = ComboInput(feat, emb_dim, None, True)
        wide, deep = layer(inputs)
        print(wide)
        print(deep)
        wide_shape = np.ones([3])
        deep_shape = np.ones([3, emb_dim])
        self.assertShapeEqual(wide_shape, wide)
        self.assertShapeEqual(deep_shape, deep)

    def test_attention_sequence_pooling_input(self):
        feat = [{"input_names": "tag_brand_list", "hash_bucket_size": 100},
                {"input_names": "tag_cate_list", "hash_bucket_size": 100},
                {"input_names": "tag_xxxx_list", "hash_bucket_size": 100}]
        input_attributes = {"tag_brand_list": {"field_type": "TagFeature", "index_type": "int", "pad_num": 5},
                            "tag_cate_list": {"field_type": "TagFeature", "index_type": "int", "pad_num": 5},
                            "tag_xxxx_list": {"field_type": "TagFeature", "index_type": "int", "pad_num": 5}}
        emb_dim = 7
        layer = AttentionSequencePoolingInput(feat, input_attributes, emb_dim)
        query = tf.constant([[3, 4, 5], [5, 6, 8]])  # feat_num = 3
        query_emb = Embedding(10, emb_dim)(query)
        query_emb = tf.reshape(query_emb, shape=[-1, 1, 3 * emb_dim])
        keys = {"tag_brand_list": {"index": tf.constant([[1, 2, 3, 0, 0], [0, 0, 0, 0, 0]]), "value": tf.constant([[1., 1., 1., 0., 0.], [0., 0., 0., 0., 0.]])},
                "tag_cate_list":  {"index": tf.constant([[1, 2, 3, 0, 0], [0, 0, 0, 0, 0]]), "value": tf.constant([[1., 1., 1., 0., 0.], [0., 0., 0., 0., 0.]])},
                "tag_xxxx_list":  {"index": tf.constant([[1, 2, 3, 0, 0], [0, 0, 0, 0, 0]]), "value": tf.constant([[1., 1., 1., 0., 0.], [0., 0., 0., 0., 0.]])},
                }
        output = layer([query_emb, keys])
        output_shape = np.ones([2, 1, 3 * emb_dim])
        print(output)
        self.assertShapeEqual(output_shape, output)


if __name__ == '__main__':
    tf.test.main()
