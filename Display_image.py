import Read_image
import image_chan
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

N_CLASSES = Read_image.N_CLASSES
train_df =Read_image.train_df
def show_class_distribution(classIDs, title):
    """
    画直方图显示样本集中交通信号的分布情况
    """
    plt.figure(figsize=(15, 5))
    plt.title('Class ID distribution for {}'.format(title))
    plt.hist(classIDs, bins=N_CLASSES)   #画直方图
    plt.show()


show_class_distribution(train_df['ClassId'], 'Train Data') #调取画图函数，绘制样本集中交通信号的分布情况



sign_name_df = pd.read_csv('signnames.csv', index_col='ClassId')    #读取signnames.csv文件并存储数据
print(sign_name_df.head(n = 5))          #查看数据前5行
sign_name_df['Occurence'] = [sum(train_df['ClassId']==c) for c in range(N_CLASSES)]#计算每一个图标的出现数量
print(sign_name_df.sort_values('Occurence', ascending=False))

SIGN_NAMES = sign_name_df.SignName.values  #获取交通标志名称，存储在SIGN_NAMES。后面程序会多次用到。
print(SIGN_NAMES[2])                  #查看ClassId为2的交通标志名称


#def change_image(train_df,SIGN_NAMES):





def load_image(image_file):
    """
    读取图片
    """
    return  plt.imread(image_file)  # 读取图片


def get_samples(image_data, num_samples, class_id=None):
    """
    随机抽取图片函数
    函数输出：随机抽取的图片的[路径，分类]信息
    函数输入：
    image_data，全部的图片文件
    num_samples，随机抽取的图片个数
    class_id=None时候不指定图片中标志分类，否则将按照指定的标志随机选取图片
    """
    if class_id is not None:
        image_data = image_data[image_data['ClassId'] == class_id]  # 指定固定的交通标志ClassId
    indices = np.random.choice(image_data.shape[0], size=num_samples, replace=False)  # 随机抽取num_samples个图片
    return image_data.iloc[indices][['Filename', 'ClassId']].values  # 输出数据格式['Filename', 'ClassId']组成的array


def show_images(image_data, cols=5, sign_names=None, show_shape=False):
    """
    Given a list of image file paths, load images and show them.
    显示给定路径的图片，路径通过get_samples函数获得
    """
    num_images = len(image_data)
    rows = num_images // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, (image_file, label) in enumerate(image_data):
        #     (image_file, label)对应['Filename', 'ClassId']，后面有用到label显示交通标志名称。
        #     enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
        #     同时列出数据和数据下标，一般用在 for 循环当中。
        #print("===================",type(image_file))
        image = load_image(image_file)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)

        if sign_names is not None:  # 显示signl名称,在原图片的左上角
            plt.text(0, 0, '{}: {}'.format(label, sign_names[label]), color='k', backgroundcolor='c', fontsize=8)

        if show_shape:  # 显示图片shape，在原图片的左下角
            plt.text(0, image.shape[0], '{}'.format(image.shape), color='k', backgroundcolor='y', fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.show()

sample_data = get_samples(train_df, 20)
show_images(sample_data, sign_names=SIGN_NAMES, show_shape=True)
#sign_names,用来显示数据label


print(SIGN_NAMES[3])
show_images(get_samples(train_df, 30, class_id=18), cols=10, show_shape=True)

def resize_image(image_file, shape=(32,32)):
    """
    函数，读取图片并调整图片尺寸为（32，32）
    输入：图片文件地址
    输出：np.array存储的调整后的图片文件
    """
    image_list=[]
    for image_file_n in image_file:
        image_file_n
        image=load_image(image_file_n)
        image=cv2.resize(image, shape)
        image_list.append(image)
    image=np.array(image_list)
    return image
X = resize_image(train_df['Filename'].values) #读取图片并调整尺寸
y = train_df['ClassId'].values                #读取分类label数据
#X,y = image_chan.img_chan(train_df)
X, y = shuffle(X, y)                          #元素随机排序sklearn.utils.shuffle
print('X data', len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=0)
#通过train_test_split函数，切割得到训练集和测试集

print('X_train:', len(X_train))
print('X_valid:', len(X_test))

