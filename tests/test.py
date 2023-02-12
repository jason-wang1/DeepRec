import tensorflow as tf
t = tf.reshape(tf.range(12), shape=[3, 4])
t2 = tf.concat([t], axis=-1)
print(t2)