import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype("float")
# print(x_data.shape)  (100,)
y_data = x_data * 0.1 + 0.3

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros(1))
# print(W) <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>

y_train = W * x_data + b

loss = tf.reduce_mean(tf.square(y_train - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
