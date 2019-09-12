import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

sess = tf.compat.v1.InteractiveSession()

I_matrix = tf.eye(5)
#print(I_matrix.eval())

X = tf.Variable(tf.eye(10))
X.initializer.run()
#print(X.eval())

A = tf.Variable(tf.random.normal([5,10]))
A.initializer.run()
#print(A.eval())

product = tf.matmul(A,X)
#print(product.eval())

B = tf.Variable(tf.random.uniform([5,10],0,2,dtype=tf.int32))
B.initializer.run()
print(B.eval())

B_new = tf.cast(B,dtype=tf.float32)
