#coding=utf-8

# Sparse Logistic Regression
import tensorflow as tf
import numpy as np
from numpy.random import RandomState
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score
import sys

if len(sys.argv) < 3:
  print "Usage:%s train_data test_data"%(sys.argv[0])
  exit(1)

fea_num = 10000000

def read_data(file_name):
  X = []
  Y = []
  with open(file_name, "r") as fin:
    for raw_line in fin:
      line = raw_line.strip(" \r\n").split(" ")
      X_i = [ int(x) for x in line[1:]]
      X.append(X_i)
      Y.append(int(line[0]))
  Y = np.reshape(np.array(Y), [-1])
  X = libsvm_2_coo(X, [len(X), fea_num]).tocsr()
  return X, Y

def shuffle(data):
    X, y = data
    ind = np.arange(X.shape[0])
    for i in range(7):
        np.random.shuffle(ind)
    return X[ind], y[ind]

def libsvm_2_coo(libsvm_data, shape):
  coo_rows = []
  coo_cols = []
  coo_data = []
  n = 0
  for x in libsvm_data:
    coo_rows.extend([n] * len(x))
    coo_cols.extend(x)
    coo_data.extend([1] * len(x))
    n+=1
  coo_rows = np.array(coo_rows)
  coo_cols = np.array(coo_cols)
  coo_data = np.array(coo_data)
  return coo_matrix((coo_data, (coo_rows, coo_cols)), shape)

def csr_2_input(csr_mat):
  coo_mat = csr_mat.tocoo()
  indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
  values = csr_mat.data
  shape = csr_mat.shape
  return indices, values, shape

def slice(csr_data, start=0, size=-1):
  if size == -1 or start+size >= csr_data[0].shape[0]:
    slc_data = csr_data[0][start:]
    slc_label = csr_data[1][start:]
  else:
    slc_data = csr_data[0][start:start + size]
    slc_label = csr_data[1][start:start + size]
  return csr_2_input(slc_data), slc_label

x = tf.sparse_placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
#weight = tf.Variable(tf.random_normal([fea_num, 1]), dtype = tf.float32)
weight = tf.Variable(tf.zeros([fea_num, 1]), dtype = tf.float32)
xw = tf.sparse_tensor_dense_matmul(x, weight)
y = tf.sigmoid(xw)

#cross entropy作为损失函数
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=xw)) \
    + tf.contrib.layers.l2_regularizer(0.001)(weight)
#创建一个训练算法 
learning_rate = 1
train_step = \
   tf.train.FtrlOptimizer(learning_rate).minimize(cross_entropy)

with tf.Session() as sess:
  #初始化所有的variable
  sess.run(tf.initialize_all_variables())

  #read data
  train_data = read_data(sys.argv[1])
  test_data = read_data(sys.argv[2])
  train_data = shuffle(train_data)
  test_data = shuffle(test_data)

  X_te, Y_te = slice(test_data)
  X_tr, Y_tr = slice(train_data)

  dataset_size = 10000
  minibatch_size = 100
  STEP = 10000
  for i in xrange(STEP):
    start = (i * minibatch_size) % dataset_size
    X, Y = slice(train_data, start, minibatch_size)
    sess.run(train_step, feed_dict={x:X, y_:Y})
  
    if i % 100 == 0:
      test_pred = sess.run(y, feed_dict={x:X_te})
      train_pred = sess.run(y, feed_dict={x: X_tr})
      test_auc =  roc_auc_score(Y_te, test_pred)
      train_auc = roc_auc_score(Y_tr, train_pred)
      print "After %d train step(s), training data %d, test auc is %g, train auc is %g"%(i, i*minibatch_size, test_auc, train_auc)

