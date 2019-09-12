import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

message = tf.constant("Welcome to the exciting world of Deep Neural Networks!")
v_1 = tf.constant([1,2,3,4])
v_2 = tf.constant([2,3,4,5])
v_add = tf.add(v_2,v_1)
v_zero = tf.zeros([4,3])
range_t = tf.linspace(0.,10.,4)
x = tf.placeholder("float")
y = 2 * x
data = tf.random_uniform([4,5],10)


with tf.compat.v1.Session() as sess:
    print(sess.run(message).decode())
    x_data = sess.run(data)
    print(sess.run(y,feed_dict={x:x_data}))