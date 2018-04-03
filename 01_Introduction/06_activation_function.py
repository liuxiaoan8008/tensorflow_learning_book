import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

x_vals = np.linspace(start=-10., stop=10., num=100)
# print x_vals

# relu
# print sess.run(tf.nn.relu([-3,3,19]))
y_relu = sess.run(tf.nn.relu(x_vals))

# relu-6
# print sess.run(tf.nn.relu6([-3,3,6,19]))
y_relu6 = sess.run(tf.nn.relu6(x_vals))

# sigmoid
# print sess.run(tf.nn.sigmoid([-1,-0.4,0,0.5,1]))
y_sigmoid = sess.run(tf.nn.sigmoid(x_vals))

# tanh
# print sess.run(tf.nn.tanh([-1,-0.4,0,0.5,1]))
y_tanh = sess.run(tf.nn.tanh(x_vals))

plt.plot(x_vals, y_relu, 'b:', label='ReLU', linewidth=2)
plt.plot(x_vals, y_relu6, 'g-.', label='ReLU6', linewidth=2)
plt.legend(loc='upper left')
plt.show()

plt.plot(x_vals, y_sigmoid, 'r--', label='Sigmoid', linewidth=2)
plt.plot(x_vals, y_tanh, 'b:', label='Tanh', linewidth=2)
plt.legend(loc='upper left')
plt.show()
