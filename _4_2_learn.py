#coding=utf-8
'''
  神经网络解决数字识别问题
'''

import tensorflow as tf
from numpy.random import RandomState
from tensorflow.examples.tutorials.mnist import input_data

#获得神经网络每一层边上的权重，
#并且把这个权重加上l2正则并加入在losses的collect集合中
def get_weight(shape, lamb):
  #生成一个变量
  var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
  #add to collection将生成的l2正则化加到集合中去
  tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lamb)(var))
  return var

#input
x = tf.placeholder(tf.float32, shape=(None, 784))
y_ = tf.placeholder(tf.float32, shape=(None, 10))

#minibatch size
batch_size = 8

#定义每一个网络节点的个数
layer_dimension = [784, 10, 10, 10, 10]
in_dimsension = layer_dimension[0]
cur_layer = x
#神经网络的层数 
n_layer = len(layer_dimension)
#构建一个神经网络
for i in xrange(1, n_layer):
  # layer_dimsension[i]是下一层网络隐层神经元个数 
  out_dimsension = layer_dimension[i]
  weight = get_weight([in_dimsension, out_dimsension], 0.001)
  biase = tf.Variable(tf.constant(0.1, shape=[out_dimsension]))
  cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight)+biase)
  in_dimsension = layer_dimension[i]

#cross entropy作为损失函数
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)

#加入损失集合
tf.add_to_collection("losses", cross_entropy_mean)
loss = tf.add_n(tf.get_collection("losses"))

#随机数据集
dataset_size = 100000
rdm = RandomState(0)
X = rdm.rand(dataset_size, layer_dimension[0])
Y = [[x1**3 + x1**2 + x2**2 + x1*x2] for (x1,x2) in X]

#创建一个训练算法 
learning_rate = 0.001
train_step = \
  tf.train.AdamOptimizer(learning_rate).minimize(loss)

#创建一个session
sess = tf.Session()
#初始化所有的variable
sess.run(tf.initialize_all_variables())

minibatch_size = 100
STEP = 100000
for i in xrange(STEP):
  start = (i*minibatch_size) % dataset_size
  end   = min(start+minibatch_size, dataset_size)

  sess.run(train_step, feed_dict={x:X[start:end],y_:Y[start:end]})
  
  #计算全部数据的loss
  if i % 1000 == 0:
    total_mese_loss = sess.run(loss, feed_dict={x:X,y_:Y})
    print "After %d train step(s), total mse on all data is %g"%(i, total_mese_loss)

sess.close()

