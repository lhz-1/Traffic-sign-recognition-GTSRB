import cv2
import matplotlib.pyplot as plt
from math import *
import numpy as np
from Read_image import N_CLASSES ,train_df

def show_class_distribution(classIDs, title):
    """
    画直方图显示样本集中交通信号的分布情况
    """
    plt.figure(figsize=(15, 5))
    plt.title('Class ID distribution for {}'.format(title))
    plt.hist(classIDs, bins=N_CLASSES)   #画直方图
    plt.show()


def change(img,a):

    height, width,_ = img.shape   #获取图片尺寸

    degree = a * 90    # 设置旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  #平移操作
    matRotation[1, 2] += (heightNew - height) / 2  # 平移操作

    newimg = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    newimg = cv2.resize(newimg,(32,32))  # 对旋转后的图片进行调整
    return newimg

def load_image(image_file):
    """
    读取图片
    """
    return  plt.imread(image_file)

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




def img_chan(train_df):
    image_list = []
    image_da =[]
    chan_da = 800
    #print(type(train_df))
    image_file = train_df['Filename'].values
    for image_file_n in image_file:
        image_file_n
        image=load_image(image_file_n)    #读取图片
        image = cv2.resize(image, (32,32))   #对读取的图片尺寸进行调整
        image_list.append(image)        #########获取image 添加到 image_list
    #print(image_list[0])
    y_train = train_df['ClassId'].values   #获取y_train 的值
    # print("==============",y_train.shape)
    # print(N_CLASSES)
    # class_indices = np.where(y_train == 0)
    # n_samples = len(class_indices[0])
    # for i in range(800 - n_samples):
    #     n = class_indices[0][i % n_samples]
    ###############     print(n)
    for class_n in range(N_CLASSES):
        class_indices = np.where(y_train == class_n)  #寻找每一类的下标数组
        n_samples = len(class_indices[0])#获取每一类的数量
    #print(n_samples)
        if n_samples < chan_da:                    #对少于1000的类进行扩充

            image_data = get_samples(train_df, n_samples, class_id=class_n)
            for i, (image_file, label) in enumerate(image_data):
                image_da.append(image_file)             #扩充数据前获得图片路径信息
            for i in range(chan_da - n_samples):
                #print(class_indices[0][i % n_samples])
                img = load_image(image_da[i % n_samples])   #获取图片
                new_img = change(img , ceil(i/n_samples))
                image_list.append(new_img)
                y_train = np.concatenate((y_train, [class_n]), axis=0)

    image_list = np.array(image_list)
    y_train = np.array(y_train)
    print('X shapes:',len(image_list))
    print('y shapes:',len(y_train))
    print(y_train)
    show_class_distribution(y_train, 'Train Data')  #调用画图函数绘制分布情况
    return image_list ,y_train

#img_chan(train_df)