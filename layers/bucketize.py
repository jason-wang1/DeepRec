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
