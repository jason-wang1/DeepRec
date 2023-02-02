l = ["", ""]
if isinstance(l, list):
    print("l is list")
if isinstance(l, str):
    print("l is str")

import tensorflow as tf
import numpy as np
print(tf.__version__)
from tensorflow.python.keras.layers import Embedding, IntegerLookup, StringLookup, Hashing, CategoryCrossing

print("split")
t = tf.constant(["7205|4377|4377|4377", "8057|6421|5239|7565|6423"])
t2 = tf.strings.split(t, '|')
layer = Hashing(num_bins=5)
emb_layer = Embedding(5, 3, embeddings_initializer='glorot_normal')
t3 = layer(t2)
t4 = emb_layer(t3)
t5 = tf.reduce_sum(t4, axis=1)
t6 = tf.reduce_sum(t5, axis=1)
print(t3)

print("Variable")
w = tf.reshape(tf.range(8, dtype=tf.float32), [2, 4, 1])
print(w)
print(tf.nn.softmax(w, axis=1))

print("embedding")
layer = Embedding(100, 1, embeddings_initializer='glorot_normal')
data = tf.constant([[1, 3, 8, 2], [1, 3, 8, 2]])
print(layer(data))

print("int lookup\n")
vocab = [12, 36, 1138, 42]
data = tf.constant([[12, 36, 1138, 42], [-2, -1, 0, 1000]])
layer = IntegerLookup(vocabulary=vocab)
print(layer.get_vocabulary())
print(layer(data))


print("str lookup\n")
vocab = ["12", "36", "1138"]
data = tf.constant([["12", "36", "1138"], ["", "0", "1000"]])
layer = StringLookup(vocabulary=vocab)
print(layer.get_vocabulary())
print(layer(data))


print("str_hash\n")
layer = Hashing(num_bins=3)
data = tf.constant([["12", "36", "1138"], ["", "0", "C"]])
print(layer(data))


print("int_hash\n")
layer = Hashing(num_bins=3)
data = tf.constant([[12, 36, 1138], [-1, 0, 1]])
print(layer(data))


print("Bucketize\n")
data = tf.constant([[-5, 10000], [150, 10], [5, 100]])
boundaries = [0, 10, 100]
layer = tf.raw_ops.Bucketize(input=data, boundaries=boundaries)
print(layer)


print("cross")
data1 = tf.constant([-50, 25, 3, 150, 8, 5])
data2 = tf.constant([100, 10, 3, 150, 1, 5])
data3 = tf.constant([100, 10, 3, 150, 1, 5])
layer = CategoryCrossing()
print(layer([data1, data2]))
print(layer([data1, data2, data3]))


print("cross2")
data1 = tf.constant([[-50, 25, 3], [150, 8, 5]])
data2 = tf.constant([[100, 10, 3], [150, 1, 5]])
data3 = tf.constant([[100, 10, 3], [150, 1, 5]])
layer = CategoryCrossing()
print(layer([data1, data2]))
print(layer([data1, data2, data3]))
