import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Embedding


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
    def __init__(self, keep_wide, input_dim, emb_dim, features_config, pad_num, reg, **kwargs):
        self.keep_wide = keep_wide
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.features_config = features_config
        self.pad_num = pad_num
        self.reg = reg
        super(InputToWideEmbV2, self).__init__(**kwargs)

    def build(self, input_shape):
        # 所有特征域的特征id被全局编码
        self.this_input_shape = input_shape
        self.emb = Embedding(self.input_dim, self.emb_dim, embeddings_regularizer=self.reg, name=f'emb')
        if self.keep_wide:
            self.wide_weight = self.add_weight(
                name='wide', shape=[self.input_dim], initializer='glorot_normal', regularizer=self.reg, trainable=True)
        super(InputToWideEmbV2, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        embedding_list = []
        wide_list = []
        for feat in self.features_config:
            if feat["feature_type"] == "ComboFeature":
                tensor = [inputs[e] for e in feat["input_names"]]
            else:
                tensor = inputs[feat["input_names"]]  # (batch_size,)  tag: (batch_size, None)  weight_tag: (batch_size, feat_num)
            if feat["feature_type"] == "WeightTagFeature":
                feat_id = tensor["index"]
                weight = tensor["value"]
                if self.keep_wide:
                    wide = tf.gather(self.wide_weight, feat_id)
                    wide = tf.reduce_sum(wide, axis=1)
                tensor = self.emb(feat_id) * tf.expand_dims(weight, axis=2)  # (batch_size, feat_id_num, emb_size)
                tensor = tf.reduce_sum(tensor, axis=1)  # (batch_size, emb_size)
            else:
                if self.keep_wide:
                    wide = tf.gather(self.wide_weight, tensor)  # (batch_size)
                tensor = self.emb(tensor)
                if feat["feature_type"] == "TagFeature":
                    tensor = tf.reduce_sum(tensor, axis=1)
                    if self.keep_wide:
                        wide = tf.reduce_sum(wide, axis=1)

            embedding_list.append(tensor)  # (batch_size, emb_size)
            if self.keep_wide:
                wide_list.append(wide)  # (batch_size,)
        emb_tensor = tf.stack(embedding_list, axis=1)  # (batch_size, feat_size, emb_size)
        if self.keep_wide:
            wide_tensor = tf.stack(wide_list, axis=1)  # (batch_size, feat_size)
            return wide_tensor, emb_tensor
        else:
            return emb_tensor
