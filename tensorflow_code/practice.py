import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 当 a 在默认图（graph）以外定义时，assert断言会报错，意味着此时a不属于默认的图
# 当 a 在默认图（graph）以内定义时，断言才是对的，说明默认会话只能在会话内部定义相关的张量（tensor）与操作（operation）
# a = tf.constant('2') ===> 默认图（graph）以外
g = tf.Graph()
with g.as_default():
    # 默认图（graph）以内
    # 此作用域内就只是一个图，在此处定义一些op（操作）和tensor（张量）
    # 区别于会话，要使用会话，则 g.Session() 才是会话， g 就是一个图（graph）
    a = tf.constant('2')
    # print(a.eval())
    # assert 断言，当判断的语句出错时，会报错，如果正常运行，则可以顺利执行而不产生任何现象
    assert a.graph is g

# a = tf.constant("5")
# b = tf.constant("6")
# sum1 = tf.add(a, b)

plt = tf.placeholder(tf.float32, [2, 3])

# with tf.Session(graph=g) as sess: ==> 设置使用的图是 g
with tf.Session() as sess:
    # print(sess.run(a))
    # print(sess.run(b))
    # print(sess.run(sum1))
    # print(tf.get_default_graph)
    # print(a.graph)
    # print(b.graph)
    # print(sum1.graph)
    # print(sess.graph)

    # 运行的是默认图中的张量
    # print(sess.run(a))

    print(sess.run(plt, feed_dict={plt: [[1, 2, 3], [4, 5, 6]]}))

