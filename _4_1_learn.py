#coding=utf-8
#搭建一个多层神经网络结构

import tensorflow as tf
from numpy.random import RandomState

#获得神经网络每一层边上的权重，并且把这个权重加上l2正则并加入在losses的
#collect集合中
def get_weight(shape, lambda):
  #生成一个变量
  var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
  #add to collection将生成的l2正则化加到集合中去
  tf.add_to_collection("losses", tf.crontrib.layers.l2_regularizer(lambda)(var))
  return var

#input
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

#minibatch size
batch_size = 8

#定义每一个网络节点的个数
layer_dimension = [2, 10, 10, 10, 1]
in_dimsension = layer_dimension[0]
cur_layer = x
#神经网络的层数 
n_layer = len(layer_dimension)
#构建一个神经网络
for i in xrange(1, n_layer):
  # layer_dimsension[i]是下一层网络隐层神经元个数 
  out_dimsension = layer_dimension[i]
  weight = get_weight([in_dimsension, out_dimsension], 0.001)
  bias = tf.Variabel(tf.constant(0.1, shape=[out_dimsension]))
  cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight)+biase)
  in_dimsension = layer_dimension[i]

#均方误差
mse_loss = tf.reduce_mean(tf.square(cur_layer - y_))
#将均方误差加入损失集合
tf.add_to_collection("losses", mse_loss)
loss = tf.add_n(tf.get_collection("losses"))
