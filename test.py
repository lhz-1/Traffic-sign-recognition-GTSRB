import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
from Traffic_Model import Model
import tensorflow as tf
import random

BATCH_SIZE = 1
N_CLASS = 43
image_size = 32  # 输入层图片大小
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "D:/projects/Traffic-Sign-Classify/model/"
MODEL_NAME = "traffic net"
new_image_path = "D:/projects/Traffic-Sign-Classify/new_image/"


sign_name_df = pd.read_csv('signnames.csv', index_col='ClassId') #读取signnames.cvs
print(sign_name_df)
# 加载需要预测的图片
def Read_new_image():
    filelist = os.listdir(new_image_path)
    index = random.sample(range(0,len(filelist)),len(filelist))

    filename = os.path.join(new_image_path ,filelist[index[0]])
    test_image = Image.open(filename)
    print(filelist[index[0]])
    test_image = np.array(test_image)
    # plt.imshow(test_image)
    # plt.show()
    test_image = cv2.resize(test_image, (32, 32))

    return test_image
X_test = Read_new_image()

plt.imshow(X_test)
plt.show()


def evaluate_one_image(X_test):
    image = tf.cast(X_test, tf.float32)
    #转换图片格式#
    # 图片原来是三维的 [32,32, 3] 重新定义图片形状 改为一个4D 四维的 tensor
    image = tf.reshape(image, [1, 32, 32, 3])
    image = tf.cast(image, tf.float32)

    logits, layers = Model(image)
    logits = tf.nn.relu(logits)
    x = tf.placeholder(tf.float32, shape=[32,32,3])
    #返回没有用激活函数，所以在这里对结果用relu激活


    with tf.Session() as sess:
        print ("Testing")
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('./model'))
        prediction = sess.run(logits, feed_dict={x: X_test})
        max_index = np.argmax(prediction)
        for i in range(0,43):
            if (max_index == i):
                sign_name_df = pd.DataFrame(pd.read_csv('signnames.csv', index_col='ClassId'))  # 读取signnames.cvs
                print('第{}类标志,它是{}'.format(i,sign_name_df.loc[i]) % prediction[:, 0])
                global FILENAME
                FILENAME= i
evaluate_one_image(X_test)
#print(FILENAME)

