l = ["", ""]
if isinstance(l, list):
    print("l is list")
if isinstance(l, str):
    print("l is str")

import tensorflow as tf
from tensorflow.keras.layers import Dense, Multiply, Lambda, Layer
import numpy as np
print(tf.__version__)
from tensorflow.python.keras.layers import Embedding, IntegerLookup, StringLookup, Hashing, CategoryCrossing

x1 = Dense(8)(np.arange(10).reshape(5, 2))
x2 = Dense(8)(np.arange(10, 20).reshape(5, 2))
multiplied = Multiply()([x1, x2])

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


print("split2")
t = tf.constant(["3879187:0.98095|3957588:1.67391|3897513:0.73381", "3869383:0.84715|3928559:1.09861", ""])
def_val = tf.constant(["0:0", "0:0", "0:0"])
t = tf.where(tf.equal(t, ""), "0:0", t)
emb_layer = Embedding(10000000, 5)
layers = [
    Lambda(lambda x: tf.strings.split(tf.strings.split(x, "|"), ":")),
    Lambda(lambda x: x.to_tensor(shape=[None, 2, 2], default_value='0')),
    Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=2)),
    Lambda(lambda x: [tf.strings.to_number(x[0], out_type=tf.int32), tf.strings.to_number(x[1], out_type=tf.float32)]),
    Lambda(lambda x: [tf.squeeze(x[0], axis=2), x[1]]),
]
for lay in layers:
    t = lay(t)
emb = emb_layer(t[0])
print("split3")
t = tf.constant(["9027227|9075427|9057945", "9115789|9027881", "0"])
layers = [
    Lambda(lambda x: tf.strings.to_number(tf.strings.split(x, "|"), out_type=tf.int32)),
]
for lay in layers:
    t = lay(t)
t = emb_layer(t)
print(t)

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
