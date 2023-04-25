import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Embedding, IntegerLookup, StringLookup, Hashing, CategoryCrossing, Lambda
from layers.dnn import DNN


class BaseInputLayer(Layer):
    """
      Input shape
        - tensor with shape ``(batch_size, tag_num)`` or ``(batch_size,)``
        - RaggedTensor with shape ``(batch_size, None)``

      Output shape
        - tensor with shape: ``(batch_size, tag_num, emb_dim)`` or ``(batch_size, emb_dim)``
        - RaggedTensor with shape ``(batch_size, None, emb_dim)``
    """
    def __init__(self, feat, emb_dim, reg, keep_wide, **kwargs):
        self.feat = feat
        self.emb_dim = emb_dim
        self.feat_name = feat["feature_name"] if "feature_name" in feat else feat["input_names"]
        self.reg = reg
        self.keep_wide = keep_wide
        super(BaseInputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if "int_vocab_list" in self.feat:
            input_dim = len(self.feat["int_vocab_list"]) + 1
            self.lookup = IntegerLookup(vocabulary=self.feat["int_vocab_list"], name=f'int_vocab_{self.feat_name}')
        elif "str_vocab_list" in self.feat:
            input_dim = len(self.feat["str_vocab_list"]) + 1
            self.lookup = StringLookup(vocabulary=self.feat["str_vocab_list"], name=f'str_vocab_{self.feat_name}')
        elif "hash_bucket_size" in self.feat:
            input_dim = self.feat["hash_bucket_size"]
            self.lookup = Hashing(self.feat["hash_bucket_size"], name=f'hash_bucket_{self.feat_name}')
        elif "boundaries" in self.feat:
            input_dim = len(self.feat["boundaries"]) + 1
            boundaries = self.feat["boundaries"]
            self.lookup = Lambda(lambda x: tf.raw_ops.Bucketize(input=x, boundaries=boundaries))
        else:
            raise ValueError(f"unexpected {self.feat}")
        self.emb = Embedding(input_dim, self.emb_dim, embeddings_regularizer=self.reg, name=f'emb_{self.feat_name}')
        if self.keep_wide:
            self.wide_var = self.add_weight(
                name=f'wide_{self.feat_name}', shape=[input_dim], initializer='glorot_normal', regularizer=self.reg, trainable=True)

    def call(self, inputs, *args, **kwargs):
        index = self.lookup(inputs)
        deep = self.emb(index)
        if self.keep_wide:
            wide = tf.gather(self.wide_var, index)
            return wide, deep
        else:
            return deep


class AttentionSequencePoolingInput(Layer):
    """
      Input shape
        - a list of tensor: [query, keys, keys_length]
        - query: candidate item - 3D tensor with shape ``(batch_size, 1, m * emb_dim)``. len(feat_list) = m
        - keys: user history seq -  a list of 2D tensor with shape ``(batch_size, pad_num)``
        - keys_length: 1D tensor with shape ``(batch_size, )``

      Output shape
        - tensor with shape: ``(batch_size, 1, m * emb_dim)``.
    """
    def __init__(self, feat_list, emb_dim, reg=None, hidden_units=None, activation='sigmoid', pad_num=20, **kwargs):
        if hidden_units is None:
            hidden_units = [36, 1]
        self.pad_num = pad_num
        self.base_layer = [BaseInputLayer(feat, emb_dim, reg, False) for feat in feat_list]
        self.dnn = DNN(hidden_units, reg, activation)
        super(AttentionSequencePoolingInput, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        query_emb, keys, keys_length = inputs
        # keys_length = keys.row_lengths(axis=1)  # (batch_size,)
        keys_emb = []
        for key, base_layer in zip(keys, self.base_layer):
            keys_emb.append(base_layer(key))  # (batch_size, pad_num, emb_dim)
        keys_emb = tf.concat(keys_emb, axis=-1)  # (batch_size, pad_num, m * emb_dim)
        queries_emb = tf.tile(query_emb, [1, self.pad_num, 1])
        att_input = tf.concat([queries_emb, keys_emb, queries_emb-keys_emb, query_emb*keys_emb],
                              axis=-1)  # (batch_size, pad_num, 4 * m * emb_dim)
        att_output = self.dnn(att_input)  # (batch_size, pad_num, 1)
        att_output = tf.transpose(att_output, perm=[0, 2, 1])  # (batch_size, 1, pad_num)

        # mask
        key_masks = tf.sequence_mask(keys_length, self.pad_num)  # (batch_size, pad_num)
        key_masks = tf.expand_dims(key_masks, axis=1)  # (batch_size, 1, pad_num)
        padding = tf.ones_like(att_output) * (-2**32+1)
        att_output = tf.where(key_masks, att_output, padding)

        # scale
        att_output = tf.math.softmax(att_output, axis=-1)
        output = tf.matmul(att_output, keys_emb)  # (batch_size, 1, m * emb_dim)
        return output


class WeightTagPoolingInput(Layer):
    """
      Input shape
        - OrderedDict({"index": tensor with shape (batch_size, tag_num),
                       "value": tensor with shape (batch_size, tag_num)})

      Output shape
        - wide tensor with shape: ``(batch_size,)``
        - deep tensor with shape: ``(batch_size, emb_dim)``.
    """
    def __init__(self, feat, emb_dim, reg, keep_wide, **kwargs):
        self.keep_wide = keep_wide
        self.base_layer = BaseInputLayer(feat, emb_dim, reg, keep_wide)
        super(WeightTagPoolingInput, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        if self.keep_wide:
            wide, deep = self.base_layer(inputs['index'])
            deep = deep * tf.expand_dims(inputs['value'], axis=2)
            deep = tf.reduce_sum(deep, axis=1)
            wide = tf.reduce_sum(wide, axis=1)
            return wide, deep
        else:
            deep = self.base_layer(inputs['index'])
            deep = deep * tf.expand_dims(inputs['value'], axis=2)
            deep = tf.reduce_sum(deep, axis=1)
            return deep


class ComboInput(Layer):
    """
      Input shape
        - a list of 1D tensor with shape ``(batch_size,)``.

      Output shape
        - wide tensor with shape: ``(batch_size,)``
        - deep tensor with shape: ``(batch_size, emb_dim)``.
    """
    def __init__(self, feat, emb_dim, reg, keep_wide, **kwargs):
        feat_name = feat["feature_name"]
        self.cross = CategoryCrossing(name=f'cross_{feat_name}')
        self.squeeze = Lambda(lambda x: tf.squeeze(x, axis=1))
        self.base_layer = BaseInputLayer(feat, emb_dim, reg, keep_wide)
        super(ComboInput, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        tensor = self.cross(inputs)
        tensor = self.squeeze(tensor)
        return self.base_layer(tensor)


class InputToWideEmb(Layer):
    """
      Input shape
        - a dict: {feat_name: feat_tensor}. For instance,
        tensor shape ``(batch_size, 1)``.

      Output shape
        - a tuple: (wide_tensor, emb_tensor)
        wide_tensor shape ``(batch_size, feat_size)``
        emb_tensor  shape ``(batch_size, feat_size, emb_dim)``
    """
    def __init__(self, keep_wide, emb_dim, features_config, reg, **kwargs):
        self.keep_wide = keep_wide
        self.emb_dim = emb_dim
        self.features_config = features_config
        self.reg = reg
        super(InputToWideEmb, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers = []
        for feat in self.features_config:
            if feat["feature_type"] in {"WeightTagFeature", "TagFeature", "SingleFeature"}:
                self.layers.append(WeightTagPoolingInput(feat, self.emb_dim, self.reg, self.keep_wide))
            elif feat["feature_type"] == "ComboFeature" and isinstance(feat["input_names"], list):
                self.layers.append(ComboInput(feat, self.emb_dim, self.reg, self.keep_wide))
            else:
                raise ValueError(f"unexpected feature_type: {feat['feature_type']}")
        super(InputToWideEmb, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        embedding_list = []
        wide_list = []
        for feat, layer in zip(self.features_config, self.layers):
            if feat["feature_type"] == "ComboFeature":
                tensor = [inputs[e] for e in feat["input_names"]]
            else:
                tensor = inputs[feat["input_names"]]  # (batch_size,)
            if self.keep_wide:
                wide, deep = layer(tensor)
                wide_list.append(wide)  # (batch_size,)
            else:
                deep = layer(tensor)
            embedding_list.append(deep)  # (batch_size, emb_dim)
        emb_tensor = tf.stack(embedding_list, axis=1)  # (batch_size, feat_size, emb_dim)
        if self.keep_wide:
            wide_tensor = tf.stack(wide_list, axis=1)  # (batch_size, feat_size)
            return wide_tensor, emb_tensor
        else:
            return emb_tensor
