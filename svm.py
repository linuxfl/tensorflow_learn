#coding=utf-8

# Logistic Regression
import tensorflow as tf
from numpy.random import RandomState

#input
x = tf.placeholder(tf.float32, shape=(None, 4))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

weight = tf.Variable(tf.random_normal([4,1]), dtype = tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

a = tf.matmul(x, weight)
y = tf.add(a, b)

#minibatch size
batch_size = 8

#损失函数
alpha = 0.01
l2_norm = alpha * tf.reduce_mean(tf.square(weight))
loss = tf.reduce_mean(tf.maximum(0.,1. - y * y_)) + l2_norm

#随机数据集
dataset_size = 100000
rdm = RandomState(0)
X = rdm.rand(dataset_size, 4)
Y = [[ x1*3 + x2*5 + x3*4 + x4*6 + 6 > 1 ] for (x1,x2,x3,x4) in X]

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
    hint_loss = sess.run(loss, feed_dict={x:X,y_:Y})
    print "After %d train step(s), total cross_entropy on all data is %g"%(i, hint_loss)

sess.close()

