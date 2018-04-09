#coding=utf-8
import tensorflow as tf
a = tf.constant([1.0,2.0], name="a")
b = tf.constant([2.0,3.0], name="b")

result = a + b

print result
#产生交互式会话
sess = tf.InteractiveSession()
print result.eval()
sess.close()

#配置会话
#allow_soft_placement : GPU上的运算会根据条件自动的放到CPU上运算
#log_device_placement : 日志中会记录每个节点被安排到了那个设备上面
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
#生成配置好的session
sess = tf.Session(config=confg)
