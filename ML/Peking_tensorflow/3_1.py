import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# x = tf.Variable(tf.random.normal([100, 2], stddev=1, mean=0, seed=2019))
# y = [[tf.square(i[0]) + tf.square(i[1]) < 2] for i in x]

BATCH_SIZE = 30
seed = 2019

rdm = np.random.RandomState(seed)
x = rdm.randn(300, 2)
y = [int(x0**2 + x1**2 < 2) for (x0, x1) in x]
y = np.vstack(y).reshape(-1, 1)
y_c = [['green' if i else 'pink'] for i in y]
# print(x)
# print(y)
# print(y_c)

plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y_c))
# plt.show()


def get_weight(shape, regulairzer):
    w = tf.Variable(tf.random.normal(shape), dtype=np.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regulairzer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


x_test = tf.placeholder(tf.float32, shape=(None, 2))
y_test = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x_test, w1) + b1)
# print(y1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y_hat = tf.matmul(y1, w2) + b2

loss_mse = tf.reduce_mean(tf.square(y_test-y_hat))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))
print(tf.get_collection('losses'))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x_test: x[start:end], y_test: y[start:end]})
        if i % 2000 == 0:
            loss_mse_v = sess.run(loss_mse, feed_dict={x_test: x, y_test: y})
            print('After %d steps, loss is: %f' %(i, loss_mse_v))
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y_hat, feed_dict={x_test: grid})
    probs = probs.reshape(xx.shape)
    print(probs)
    print("w1: ", sess.run(w1))
    print("b1: ", sess.run(b1))
    print("w2: ", sess.run(w2))
    print("b2: ", sess.run(b2))

    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y_c))
    plt.contour(xx, yy, probs, levels=[0.5])  # 给 probs=0.5 的点上色
    plt.show()

