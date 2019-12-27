import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backwards
import time

TEST_INTERVAL = 5


def test_model(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape=(None, mnist_forward.INPUT_NODE))
        y_true = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])

        # 计算测试集预测值
        y_hat = mnist_forward.mnist_forward(x)

        # 恢复滑动平均值
        ema = tf.train.ExponentialMovingAverage(mnist_backwards.MOVING_AVERAGE_RATE)
        ema_restore = ema.variables_to_restore()

        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backwards.MODEL_SAVE_DIRECTORY)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels})
                    print("After %s training step(s), test accuracy is %g." % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found!!!")
                    return
            time.sleep(TEST_INTERVAL)


if __name__ == '__main__':
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    test_model(mnist)

