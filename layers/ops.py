import tensorflow as tf
from tensorflow.python.keras.layers import Layer

class Bucketize(Layer):
    """Bucketizes 'input' based on 'boundaries'.

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``.
    """
    def __init__(self, boundaries, name: str):
        self.boundaries = boundaries
        super(Bucketize, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        return tf.raw_ops.Bucketize(input=inputs, boundaries=self.boundaries)


class Split(Layer):
    """split string 'input' based on 'sep'.

      Input shape
        - tensor with shape: ``(batch_size, 1)``.

      Output shape
        - RaggedTensor with shape: ``(batch_size, None)``.
    """
    def __init__(self, sep, **kwargs):
        self.sep = sep
        super(Split, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.strings.split(input=inputs, sep=self.sep)


class ReduceSum(Layer):
    def __init__(self, axis, keepdims, **kwargs):
        self.axis = axis
        self.keepdims = keepdims
        super(ReduceSum, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.reduce_sum(input=inputs, axis=self.axis, keepdims=self.keepdims)
