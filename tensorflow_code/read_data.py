import tensorflow as tf
import os


def read_csv(file_list):
    """
    读取csv文件
    :param file_list:
    :return: None
    """
    # print(file_list)
    # 构造文件队列
    # file_queue = tf.train.string_input_producer(file_list)

    # 构造csv阅读器来读取数据
    # 这个API已经被废弃掉了
    # reader = tf.TextLineReader()
    # key, value = reader(file_queue)
    # print(value)

    res = tf.data.TextLineDataset(file_list)
    # print(res)

    return None


if __name__ == '__main__':
    # 构建文件队列
    file_list = os.listdir('./data/csv')
    # print(file_list)
    # read_csv(file_list)
    dataset = tf.data.TextLineDataset(file_list)
    # dataset = tf.data.from
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    init_op = iterator.initializer

    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(next_element))
