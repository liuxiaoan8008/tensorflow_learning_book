#!usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session() # 开启一个会话

row_dim = 2
col_dim = 3 

# 定义变量
rand_var1 = tf.get_variable('rand_var1', initializer = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0))
rand_var2 = tf.get_variable('rand_var2', initializer = tf.random_uniform([row_dim, col_dim],minval=0, maxval=4))

merged = tf.summary.merge_all() # 将所有的tensor添加到tensorboard
writer = tf.summary.FileWriter('/tmp/tf/variable_logs', graph=sess.graph) # 初始化一个graph writer，用于保存计算图。

sess.run(tf.global_variables_initializer())
print sess.run(rand_var1)
print sess.run(rand_var2)

