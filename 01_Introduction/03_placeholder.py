#!usr/bin/env pyhton
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session() 

# 我们通过tf.placeholder(..)方法来定义placeholder，这个方法需要接受两个参数：data-type 和data-shape。直接看代码。
x = tf.placeholder(tf.float32, shape=[4,4])

# 为了说明placeholder的使用，我们构造一个数据，然后将其传入placeholder中。
data = np.random.rand(4,4)

print sess.run(x, feed_dict={x: data}) # 通过feed_dict参数来指定对应的feed数据

# 可视化一波
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("/tmp/tf3/variable_logs", sess.graph)

