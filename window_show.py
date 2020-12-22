from playsound import playsound
import sys
import time
from PyQt5.QtCore import QObject, pyqtSignal, QEventLoop, QTimer
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from skimage import io




import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
from Traffic_Model import Model
import tensorflow as tf
import random


'''
控制台输出定向到Qtextedit中
'''


class Stream(QObject):
    """Redirects console output to text widget."""
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))


class GenMast(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.initUI()

        # Custom output stream.
        sys.stdout = Stream(newText=self.onUpdateText)

    def onUpdateText(self, text):
        """Write console output to text widget."""
        cursor = self.process.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()
    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

    def initUI(self):



        """Creates UI window on launch."""
        # Button for generating the master list.
        btnvidio =QPushButton('打开摄像头', self) #摄像头按钮
        btnvidio.move(550, 15)
        btnvidio.resize(100, 25)
        btnvidio.clicked.connect(self.vidio_Clicked)


        btnGenMast = QPushButton('加载并预测', self) #按钮
        btnGenMast.move(550, 50)
        btnGenMast.resize(100,25)
        btnGenMast.setToolTip('选择一张图片，并进行预测')
        btnGenMast.clicked.connect(self.genMastClicked)   #槽函数

        # read_but = QPushButton("保存图片",self)
        # read_but.resize(100,25)
        # read_but.move(350, 20)
        # read_but.clicked.connect(self.opendir)



        # Create the text output widget.
        self.process = QTextEdit(self, readOnly=True)
        self.process.ensureCursorVisible()
        self.process.setLineWrapColumnOrWidth(500)
        self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)
        self.process.setFixedWidth(400)
        self.process.setFixedHeight(200)
        self.process.move(25, 250)
        self.label1 = QLabel(self)
        self.label1.setText("   显示图片")     #显示图片窗口i
        self.label1.setFixedSize(200, 200)
        self.label1.move(25, 10)
        self.label1.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}")

        # Set window size and title, then show the window.
        self.setGeometry(300, 100, 700, 500)
        self.setWindowTitle('预测识别')                     #主窗口
        #self.setWindowIcon(QIcon('D:/projects/Traffic-Sign-Classify/window_material/ico/小车.ico'))
        self.setObjectName("MainWindow")
        self.setStyleSheet("#MainWindow{border-image:url(D:/projects/Traffic-Sign-Classify/window_material/window_backgrangd.jpg)}")  # 这里使用相对路径，也可以使用绝对路径
        #self.setStyleSheet('D:/projects/Traffic-Sign-Classify/window_material/window_backgrangd.jpg')                      #设置窗口背景
        self.show()


    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", r"D:/projects/Traffic-Sign-Classify/new_image/", "*.*")
        jpg = QtGui.QPixmap(imgName).scaled(self.label1.width(), self.label1.height())
        self.label1.setPixmap(jpg)
        test_image = Image.open(imgName)
        test_image = np.array(test_image)
        plt.imshow(test_image)
        plt.show()
        test_image = cv2.resize(test_image, (32, 32))

        return test_image


    def printhello(self):

        BATCH_SIZE = 1
        N_CLASS = 43
        image_size = 32  # 输入层图片大小
        # 模型保存的路径和文件名
        MODEL_SAVE_PATH = "D:/projects/Traffic-Sign-Classify/model/"
        MODEL_NAME = "traffic net"
        new_image_path = "D:/projects/Traffic-Sign-Classify/new_image/"

        sign_name_df = pd.read_csv('signnames.csv', index_col='ClassId')  # 读取signnames.cvs
        print(sign_name_df)

        # 加载需要预测的图片

        X_test = self.openimage()

        plt.imshow(X_test)
        plt.show()

        def evaluate_one_image(X_test):
            tf.reset_default_graph()
            image = tf.cast(X_test, tf.float32)
            # 转换图片格式#
            # 图片原来是三维的 [32,32, 3] 重新定义图片形状 改为一个4D 四维的 tensor
            image = tf.reshape(image, [1, 32, 32, 3])
            image = tf.cast(image, tf.float32)

            logits, layers = Model(image)
            logits = tf.nn.relu(logits)
            x = tf.placeholder(tf.float32, shape=[32, 32, 3])
            # 返回没有用激活函数，所以在这里对结果用relu激活

            with tf.Session() as sess:
                print("Testing")
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint('./model'))
                prediction = sess.run(logits, feed_dict={x: X_test})
                max_index = np.argmax(prediction)
                for i in range(0, 43):
                    if (max_index == i):
                        sign_name_df = pd.DataFrame(
                            pd.read_csv('signnames.csv', index_col='ClassId'))  # 读取signnames.cvs
                        print('第{}类标志,它是{}'.format(i, sign_name_df.loc[i]) % prediction[:, 0])


        evaluate_one_image(X_test)
        playsound('D:/projects/Traffic-Sign-Classify/window_material/mp3/1.mp3')

    def vidio_Clicked(self):
        """打开摄像头"""
        cap = cv2.VideoCapture(0)  # 打开摄像头
        while (1):
            # get a frame
            ret, frame = cap.read()
            # show a frame
            winvidio = cv2.imshow("capture", frame)  # 生成摄像头窗口
            key = cv2.waitKey(1)
            if key == ord('s'): # 如果按下s 就截图
                cv2.imwrite("D:/projects/Traffic-Sign-Classify/new_image/test.png", frame)  # 保存路径
            if key == ord('q'): # 如果按下q 就退出
                break


        cap.release()
        cv2.destroyAllWindows()




    def genMastClicked(self):
        """Runs the main function."""
        print('Running...')
        self.printhello()
        loop = QEventLoop()
        QTimer.singleShot(2000, loop.quit)
        loop.exec_()
        print('Done.')


if __name__ == '__main__':
    # Run the application.
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    app.setWindowIcon(QIcon('D:/projects/Traffic-Sign-Classify/window_material/ico/小车.ico'))


    gui = GenMast()
    sys.exit(app.exec_())
