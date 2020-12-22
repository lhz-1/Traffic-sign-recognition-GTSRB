from Display_image import resize_image
from Traffic_Model import Model
import glob
import pandas as pd
import cv2
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import tensorflow as tf


BATCH_SIZE = 128
rate = 0.001
TEST_IMAGE_DIR= 'D:\projects\Traffic-Sign-Classify\Data\Final_Test'
path1 = 'D:\projects\Traffic-Sign-Classify\Data\Final_Test\Images'



test_file=glob.glob(os.path.join(TEST_IMAGE_DIR, '*/GT-*.csv'))

folder = test_file[0].split('\\')[6]
test_df1 = pd.read_csv(test_file[0], sep=';')                     #读取文件夹中csv文件
test_df1['Filename'] = test_df1['Filename'].apply(lambda x: os.path.join(path1,x))#将‘Filename'列内容延展，增加文件地址
print("Number of test images: {:>5}".format(test_df1.shape[0]))

X_test = resize_image(test_df1['Filename'].values) #读取图片并调整尺寸
y_test = test_df1['ClassId'].values              #读取分类label数据

print(test_df1.head())

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

logits, layers = Model(x)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()



def evaluate(X_data, y_data):

    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./model'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))








