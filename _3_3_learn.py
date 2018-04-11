#coding=utf-8
import tensorflow as tf

#随机一个呈正态分布的矩阵，方差为2
#weight = tf.Variable(tf.random_normal([2, 3], stddev=2))
#均匀分布
#weight = tf.Varibale(tf.random_uniform([2, 3]))
#偏置，给一个0值
#biase = tf.Variable(tf.zeros([3]))
#biase = tf.Variable(tf.ones([3]))


#seed=1时每次运程序生成的随机数是一样的
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

#输入一个常量，x是一个1X2的矩阵
x = tf.constant([[2.0, 3.0]])

#前向传播算法得到神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
#初始化 w1 变量和 w2变量
#sess.run(w1.initializer)
#sess.run(w2.initializer)

sess.run(tf.initialize_all_variables())
print sess.run(y)
sess.close()
