import tensorflow as tf
import mnist_forward
import os
from tensorflow.examples.tutorials.mnist import input_data

REGULARIZER = 0.0001
LEARNING_RATE_BASE = 0.01
BATCH_SIZE = 200
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_RATE = 0.99
STEPS = 50000
MODEL_SAVE_DIRECTORY = './model/'
MODEL_NAME = "mnist_model"


def mnist_backwards(mnist):
    x = tf.placeholder(tf.float32, shape=(None, mnist_forward.INPUT_NODE))
    y_true = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])

    y_hat = mnist_forward.mnist_forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    # 学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step, mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
    )

    # 交叉熵
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=tf.argmax(y_true, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 衰减率
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_RATE, global_step)
    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step, ce_value = sess.run([train_op, loss, global_step, ce], feed_dict={x: xs, y_true: ys})

            if i % 10000 == 0:
                print("After %d training step(s), loss is %g." % (step, loss_value))
                print("cross entropy is ", ce_value)

        saver.save(sess, os.path.join(MODEL_SAVE_DIRECTORY, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./data", one_hot=True)
    mnist_backwards(mnist)


if __name__ == '__main__':
    main()
