import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Embedding, IntegerLookup, StringLookup, Hashing, CategoryCrossing, Lambda


class BaseInputLayer(Layer):
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


class WeightTagPoolingInput(BaseInputLayer):
    """
      Input shape
        - OrderedDict({"index": tensor with shape (batch_size, tag_num),
                       "value": tensor with shape (batch_size, tag_num)})

      Output shape
        - wide tensor with shape: ``(batch_size,)``
        - deep tensor with shape: ``(batch_size, emb_size)``.
    """
    def __init__(self, feat, emb_dim, reg, keep_wide, **kwargs):
        super(WeightTagPoolingInput, self).__init__(feat, emb_dim, reg, keep_wide, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        index = self.lookup(inputs['index'])
        deep = self.emb(index) * tf.expand_dims(inputs['value'], axis=2)
        deep = tf.reduce_sum(deep, axis=1)
        if self.keep_wide:
            wide = tf.gather(self.wide_var, index)
            wide = tf.reduce_sum(wide, axis=1)
            return wide, deep
        else:
            return deep


class TagPoolingInput(BaseInputLayer):
    """
      Input shape
        - 2D RaggedTensor with shape ``(batch_size, None)``

      Output shape
        - wide tensor with shape: ``(batch_size,)``
        - deep tensor with shape: ``(batch_size, emb_size)``.
    """
    def __init__(self, feat, emb_dim, reg, keep_wide, **kwargs):
        super(TagPoolingInput, self).__init__(feat, emb_dim, reg, keep_wide, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        index = self.lookup(inputs)
        deep = self.emb(index)
        deep = tf.reduce_sum(deep, axis=1)
        if self.keep_wide:
            wide = tf.gather(self.wide_var, index)
            wide = tf.reduce_sum(wide, axis=1)
            return wide, deep
        else:
            return deep


class IdInput(BaseInputLayer):
    """
      Input shape
        - 1D tensor with shape ``(batch_size,)``.

      Output shape
        - wide tensor with shape: ``(batch_size,)``
        - deep tensor with shape: ``(batch_size, emb_size)``.
    """
    def __init__(self, feat, emb_dim, reg, keep_wide, **kwargs):
        super(IdInput, self).__init__(feat, emb_dim, reg, keep_wide, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        index = self.lookup(inputs)
        deep = self.emb(index)
        if self.keep_wide:
            wide = tf.gather(self.wide_var, index)
            return wide, deep
        else:
            return deep


class RawInput(BaseInputLayer):
    """
      Input shape
        - 1D tensor with shape ``(batch_size,)``.

      Output shape
        - wide tensor with shape: ``(batch_size,)``
        - deep tensor with shape: ``(batch_size, emb_size)``.
    """
    def __init__(self, feat, emb_dim, reg, keep_wide, **kwargs):
        super(RawInput, self).__init__(feat, emb_dim, reg, keep_wide, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        index = self.lookup(inputs)
        deep = self.emb(index)
        if self.keep_wide:
            wide = tf.gather(self.wide_var, index)
            return wide, deep
        else:
            return deep


class ComboInput(BaseInputLayer):
    """
      Input shape
        - a list of 1D tensor with shape ``(batch_size,)``.

      Output shape
        - wide tensor with shape: ``(batch_size,)``
        - deep tensor with shape: ``(batch_size, emb_size)``.
    """
    def __init__(self, feat, emb_dim, reg, keep_wide, **kwargs):
        feat_name = feat["feature_name"]
        self.cross = CategoryCrossing(name=f'cross_{feat_name}')
        self.squeeze = Lambda(lambda x: tf.squeeze(x, axis=1))
        super(ComboInput, self).__init__(feat, emb_dim, reg, keep_wide, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        tensor = self.cross(inputs)
        tensor = self.squeeze(tensor)
        index = self.lookup(tensor)
        deep = self.emb(index)
        if self.keep_wide:
            wide = tf.gather(self.wide_var, index)
            return wide, deep
        else:
            return deep


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
    def __init__(self, keep_wide, emb_dim, features_config, reg, **kwargs):
        self.keep_wide = keep_wide
        self.emb_dim = emb_dim
        self.features_config = features_config
        self.reg = reg
        super(InputToWideEmb, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers = []
        for feat in self.features_config:
            if feat["feature_type"] == "WeightTagFeature":
                self.layers.append(WeightTagPoolingInput(feat, self.emb_dim, self.reg, self.keep_wide))
            elif feat["feature_type"] == "TagFeature":
                self.layers.append(TagPoolingInput(feat, self.emb_dim, self.reg, self.keep_wide))
            elif feat["feature_type"] == "IdFeature":
                self.layers.append(IdInput(feat, self.emb_dim, self.reg, self.keep_wide))
            elif feat["feature_type"] == "RawFeature":
                self.layers.append(RawInput(feat, self.emb_dim, self.reg, self.keep_wide))
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
            embedding_list.append(deep)  # (batch_size, emb_size)
        emb_tensor = tf.stack(embedding_list, axis=1)  # (batch_size, feat_size, emb_size)
        if self.keep_wide:
            wide_tensor = tf.stack(wide_list, axis=1)  # (batch_size, feat_size)
            return wide_tensor, emb_tensor
        else:
            return emb_tensor
