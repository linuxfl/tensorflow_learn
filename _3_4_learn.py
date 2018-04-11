#coding=utf-8
import tensorflow as tf
from numpy.random import RandomState

#seed=1时每次运程序生成的随机数是一样的
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name="w1")
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1), name="w2")
batch_size = 8
#定义一个placeholder
x = tf.placeholder(tf.float32, shape=(None, 2), name="input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="label")

#前向传播算法得到神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#cross entropy
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
learning_rate = 0.001
train_step = \
  tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#生成随机数
rdm = RandomState(1)
dataset_size = 128
dim = 2
X = rdm.rand(dataset_size, dim)
Y = [[int(x1+x2) < 1] for (x1,x2) in X]

sess = tf.Session()

#初始化 w1 变量和 w2变量
sess.run(tf.initialize_all_variables())

STEP=10000
print sess.run(w1)
print sess.run(w2)
for i in xrange(STEP):
  #每次取一个batch_size的数据进行训练
  start = (i*batch_size) % dataset_size
  end = min(start+batch_size, dataset_size)

  sess.run(train_step, feed_dict={x:X[start:end],y_:Y[start:end]})

  #计算全部数据的cross_entropy
  if i % 1000 == 0:
    total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X,y_:Y})
    print "After %d train step(s), cross_entropy on all data is %g"%(i, total_cross_entropy)

print sess.run(w1)
print sess.run(w2)
sess.close()
