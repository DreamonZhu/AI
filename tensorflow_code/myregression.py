import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def myrgression():
    """
    自实现一个线性规划的模型
    :return: None
    """
    # 1. 准备数据
    with tf.variable_scope('data'):
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name='x_data')
        y_true = tf.matmul(x, [[0.7]]) + 1.0
    # 2. 定义模型
    with tf.variable_scope('model'):
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0, name="w"))
        bias = tf.Variable(0.0, name="b")
    # 3. 计算误差
    with tf.variable_scope('loss'):
        y_predict = tf.matmul(x, weight) + bias
        loss = tf.reduce_mean(tf.square(y_true - y_predict))
    # 4. 梯度优化
    with tf.variable_scope('gradient'):
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    # 收集tensor
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('bias', bias)
    tf.summary.histogram('weight', weight)
    # 合并tensor
    merged = tf.summary.merge_all()

    initial_variables = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initial_variables)
        # print(sess.run(weight))
        # print(weight)
        print("初始的权重是 %s, 偏置是 %s" % (weight.eval(), bias.eval()))
        # 建立事件文件
        filewriter = tf.summary.FileWriter("./temp/summary/test/", graph=sess.graph)

        for i in range(500):
            sess.run(train_op)
            summary = sess.run(merged)
            filewriter.add_summary(summary, i)

            print("第 %s 次的权重是：%s, 偏置是：%s" % (i, weight.eval(), bias.eval()))
    return None


if __name__ == '__main__':
    myrgression()
