import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

identity_matrix = tf.diag([1.0,1.0,1.0])
sess.run(identity_matrix)

random_matrix = tf.truncated_normal([2,3]) # [-1,1]
print sess.run(random_matrix)

C = tf.random_uniform([3,2]) # [0,1]
print(sess.run(C))

matrix_np = np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]])
matrix_tf = tf.convert_to_tensor(matrix_np)
sess.run(matrix_tf)

print sess.run(C+C)
print sess.run(C-C)

