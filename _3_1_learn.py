#coding=utf-8
import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([1.0, 3.0], name="b")

#生成一个张量
result = a + b
print result
#创建一个会话，把生成的张量放入会话得到取值
#用python上下文管理器，防止意外关闭导致资源
#没有释放
with tf.Session() as sess:
  sess = tf.Session()
  print sess.run(result)

#生成一个默认会话，默认会话会把默认生成的计算
#图加入这个会话，所以不需要再调用 run方法，而是
#通过 result.eval 直接取得计算结果
with tf.Session().as_default():
  print result.eval()

sess = tf.Session()
print sess.run(result)

#等价的计算图和会话的使用，session的s要小写
print result.eval(session=sess)

#没有用with的情况下来主动关掉会话
sess.close()
