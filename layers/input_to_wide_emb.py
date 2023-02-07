import sys
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Embedding, IntegerLookup, StringLookup, Hashing, CategoryCrossing
from layers.ops import Split, Bucketize


class InputToWideEmb(Layer):
    """
      Input shape
        - a dict: {feat_name: feat_tensor}. For instance,
        tensor shape ``(batch_size, 1)``.

      Output shape
        - a tuple: (wide_tensor, emb_tensor)
        wide_tensor shape ``(batch_size, feat_size)``
        emb_tensor  shape ``(batch_size, feat_size, emb_size)``
    """
    def __init__(self, emb_dim, features_config, reg, **kwargs):
        self.emb_dim = emb_dim
        self.features_config = features_config
        self.reg = reg
        super(InputToWideEmb, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_to_emb_layer = {}
        self.wide_weight_layer = {}
        for feat in self.features_config:
            feat_name = feat["feature_name"] if "feature_name" in feat else feat["input_names"]
            self.input_to_emb_layer[feat_name] = []
            # if feat["feature_type"] == "TagFeature":
            #     self.input_to_emb_layer[feat_name].append(Split("|"))

            if feat["feature_type"] == "ComboFeature" and isinstance(feat["input_names"], list):
                self.input_to_emb_layer[feat_name].append(CategoryCrossing(name=f'cross_{feat_name}'))

            if "int_vocab_list" in feat:
                input_dim = len(feat["int_vocab_list"]) + 1
                int_lookup = IntegerLookup(vocabulary=feat["int_vocab_list"], name=f'int_vocab_{feat_name}')
                emb = Embedding(input_dim, self.emb_dim, embeddings_regularizer=self.reg, name=f'emb_{feat_name}')
                self.input_to_emb_layer[feat_name].extend([int_lookup, emb])
            elif "str_vocab_list" in feat:
                input_dim = len(feat["str_vocab_list"]) + 1
                str_lookup = StringLookup(vocabulary=feat["str_vocab_list"], name=f'str_vocab_{feat_name}')
                emb = Embedding(input_dim, self.emb_dim, embeddings_regularizer=self.reg, name=f'emb_{feat_name}')
                self.input_to_emb_layer[feat_name].extend([str_lookup, emb])
            elif "hash_bucket_size" in feat:
                input_dim = feat["hash_bucket_size"]
                hash_bucket = Hashing(feat["hash_bucket_size"], name=f'hash_bucket_{feat_name}')
                emb = Embedding(input_dim, self.emb_dim, embeddings_regularizer=self.reg, name=f'emb_{feat_name}')
                self.input_to_emb_layer[feat_name].extend([hash_bucket, emb])
            elif "boundaries" in feat:
                input_dim = len(feat["boundaries"]) + 1
                raw_bucket = Bucketize(feat["boundaries"], name=f'raw_bucket_{feat_name}')
                emb = Embedding(input_dim, self.emb_dim, embeddings_regularizer=self.reg, name=f'emb_{feat_name}')
                self.input_to_emb_layer[feat_name].extend([raw_bucket, emb])
            else:
                print(f"feature_config error, input_names: {feat['input_names']}")
                sys.exit(1)
            self.wide_weight_layer[feat_name] = self.add_weight(
                name=f'wide_{feat_name}', shape=[input_dim], initializer='glorot_normal', regularizer=self.reg, trainable=True)

        super(InputToWideEmb, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        embedding_list = []
        wide_list = []
        for feat in self.features_config:
            feat_name = feat["feature_name"] if "feature_name" in feat else feat["input_names"]
            if feat["feature_type"] == "ComboFeature":
                tensor = [inputs[e] for e in feat["input_names"]]
            else:
                tensor = inputs[feat["input_names"]]  # (batch_size,)
            for layer in self.input_to_emb_layer[feat_name]:
                if isinstance(layer, Embedding):
                    wide = tf.gather(self.wide_weight_layer[feat_name], tensor)
                    if feat["feature_type"] == "TagFeature":
                        wide = tf.reduce_sum(wide, axis=1)
                    wide_list.append(wide)  # (batch_size,)
                tensor = layer(tensor)
            if feat["feature_type"] == "TagFeature":
                tensor = tf.reduce_sum(tensor, axis=1)
            embedding_list.append(tensor)  # (batch_size, emb_size)
        wide_tensor = tf.stack(wide_list, axis=1)  # (batch_size, feat_size)
        emb_tensor = tf.stack(embedding_list, axis=1)  # (batch_size, feat_size, emb_size)
        return wide_tensor, emb_tensor
