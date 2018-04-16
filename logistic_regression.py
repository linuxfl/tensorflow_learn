#coding=utf-8

# Logistic Regression
import tensorflow as tf
from numpy.random import RandomState

#input
x = tf.placeholder(tf.float32, shape=(None, 4))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

weight = tf.Variable(tf.random_normal([4,1]), dtype = tf.float32)

y = tf.matmul(x, weight)
#minibatch size
batch_size = 8

#cross entropy作为损失函数
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 10e-10, 1)))

#随机数据集
dataset_size = 100000
rdm = RandomState(0)
X = rdm.rand(dataset_size, 4)
Y = [[ x1*3 + x2*5 + x3*4 + x4*6 + 6 > 1 ] for (x1,x2,x3,x4) in X]

#创建一个训练算法 
learning_rate = 0.001
train_step = \
  tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

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
    total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X,y_:Y})
    print "After %d train step(s), total cross_entropy on all data is %g"%(i, total_cross_entropy)

sess.close()

