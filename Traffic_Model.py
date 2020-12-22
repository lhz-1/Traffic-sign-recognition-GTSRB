import tensorflow as tf
from tensorflow.contrib.layers import flatten   #pool是全连接层的输入，则需要将其转换为一个向量。假设pool是一个100*7*7*64的矩阵，则通过转换后，得到一个[100,3136]的矩阵，这里100位卷积神经网络的batch_size，3136则是7*7*64的乘积。

def Model(x):
    # 超参数设置/Hyperparameters
    mu = 0
    sigma = 0.01
    keep_prob = 1

    # 第一层: 卷积. 输入 = 32x32x3. 输出 = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # 激活函数
    active1 = tf.nn.softplus(conv1)
    # 池化（Pooling）：输入 = 28x28x6. 输出 = 14x14x6.
    pool1 = tf.nn.max_pool(active1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 第二层: 卷积. 输出 = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # 激活函数
    active2 = tf.nn.softplus(conv2)
    # 池化（Poolin）：输入 = 10x10x16. 输出 = 5x5x16.
    pool2 = tf.nn.max_pool(active2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Flatten. 输入 = 5x5x16. 输出 = 400.
    fc0 = flatten(pool2)

    # 第三层: 全连接（Fully Connected）.输入= 400. 输出 = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    # 激活函数
    fc1 = tf.nn.softplus(fc1)

    # 第四层: 全连接（Fully Connected）. 输入 = 120. 输出 = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    # 激活函数.
    fc2 = tf.nn.softplus(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # 第五层: 全连接（Fully Connected）. 输入= 84. 输出 = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits, {'conv1': conv1, 'conv2': conv2, 'activ1': active1, 'activ2': active2, 'pool1': pool1}