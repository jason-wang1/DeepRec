import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Embedding, Lambda, CategoryCrossing


class InputToWideEmbV2(Layer):
    """
      Input shape
        - a dict: {feat_name: feat_tensor}. For instance,
        tensor shape ``(batch_size, 1)``.

      Output shape
        - a tuple: (wide_tensor, emb_tensor)
        wide_tensor shape ``(batch_size, feat_size)``
        emb_tensor  shape ``(batch_size, feat_size, emb_size)``
    """
    def __init__(self, keep_wide, input_dim, emb_dim, features_config, pref_pad_num, reg, **kwargs):
        self.keep_wide = keep_wide
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.features_config = features_config
        self.pref_pad_num = pref_pad_num
        self.reg = reg
        super(InputToWideEmbV2, self).__init__(**kwargs)

    def build(self, input_shape):
        # 所有特征域的特征id被全局编码
        self.emb = Embedding(self.input_dim, self.emb_dim, embeddings_regularizer=self.reg, name=f'emb')
        if self.keep_wide:
            self.wide_weight = self.add_weight(
                name='wide', shape=[self.input_dim], initializer='glorot_normal', regularizer=self.reg, trainable=True)

        self.input_to_emb_layer = {}
        for feat in self.features_config:
            feat_name = feat["feature_name"] if "feature_name" in feat else feat["input_names"]
            self.input_to_emb_layer[feat_name] = []
            self.input_to_emb_layer[feat_name].append(Lambda(lambda x: tf.expand_dims(x, axis=1)))
            if feat["feature_type"] == "TagFeature":
                self.input_to_emb_layer[feat_name].extend([
                    # Lambda(lambda x: tf.where(tf.equal(x, ""), "0", x)),
                    Lambda(lambda x: tf.strings.to_number(tf.strings.split(x, "|"), out_type=tf.int32)),
                ])

            elif feat["feature_type"] == "PrefFeature":
                self.input_to_emb_layer[feat_name].extend([
                    # Lambda(lambda x: tf.where(tf.equal(x, ""), "0:0", x)),
                    Lambda(lambda x: tf.strings.split(tf.strings.split(x, "|"), ":")),
                    Lambda(lambda x: x.to_tensor(shape=[None, 1, self.pref_pad_num, 2], default_value='0')),
                    Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=3)),
                    Lambda(lambda x: [tf.strings.to_number(x[0], out_type=tf.int32), tf.strings.to_number(x[1], out_type=tf.float32)]),
                    Lambda(lambda x: [tf.squeeze(x[0], axis=3), x[1]]),
                ])

            elif feat["feature_type"] == "ComboFeature" and isinstance(feat["input_names"], list):
                self.input_to_emb_layer[feat_name].append(CategoryCrossing(name=f'cross_{feat_name}'))
        super(InputToWideEmbV2, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        embedding_list = []
        wide_list = []
        for feat in self.features_config:
            feat_name = feat["feature_name"] if "feature_name" in feat else feat["input_names"]
            if feat["feature_type"] == "ComboFeature":
                tensor = [inputs[e] for e in feat["input_names"]]
            else:
                tensor = inputs[feat["input_names"]]  # (batch_size, 1)
            for layer in self.input_to_emb_layer.get(feat_name, []):
                tensor = layer(tensor)

            if feat["feature_type"] == "PrefFeature":
                feat_id, weight = tensor
                if self.keep_wide:
                    wide = tf.gather(self.wide_weight, feat_id)
                    wide = tf.reduce_sum(wide, axis=2)
                tensor = self.emb(feat_id) * weight  # (batch_size, feat_id_num, emb_size)
                tensor = tf.reduce_sum(tensor, axis=2)
            else:
                if self.keep_wide:
                    wide = tf.gather(self.wide_weight, tensor)  # (batch_size, 1)
                tensor = self.emb(tensor)
                if feat["feature_type"] == "TagFeature":
                    tensor = tf.reduce_sum(tensor, axis=2).to_tensor()
                    if self.keep_wide:
                        wide = tf.reduce_sum(wide, axis=2).to_tensor()

            embedding_list.append(tensor)  # (batch_size, 1, emb_size)
            if self.keep_wide:
                wide_list.append(wide)  # (batch_size, 1, emb_size)
        emb_tensor = tf.concat(embedding_list, axis=1)  # (batch_size, feat_size, emb_size)
        if self.keep_wide:
            wide_tensor = tf.concat(wide_list, axis=1)  # (batch_size, feat_size)
            return wide_tensor, emb_tensor
        else:
            return emb_tensor
