import glob
import pandas as pd
import cv2
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 导入数据并可视化
TRAIN_IMAGE_DIR = 'D:\projects\Traffic-Sign-Classify\Data\GTSRB\Final_Training\Images' #此处为文件夹地址
Test_IMAGE_DIR= 'D:\projects\Traffic-Sign-Classify\Data\Final_Test'
dfl = []                                                        #创建空list用于存储数据
for train_file in glob.glob(os.path.join(TRAIN_IMAGE_DIR, '*/GT-*.csv')):
    '''
    循环读取\Images文件夹下，每一个子文件夹内的csv问价并存储到dfs中
    '''
    path = os.path.split(train_file)
    path = os.path.split(path[0])
    path1 = path[1]


    folder = train_file.split('\\')[5]    #对地址进行分割得到Images文件夹下的0000/0001...子文件夹名
    df = pd.read_csv(train_file, sep=';')                     #读取文件夹中csv文件x
    df['Filename'] = df['Filename'].apply(lambda x: os.path.join(TRAIN_IMAGE_DIR, path1,x))  # 将‘Filename'列内容延展，增加文件地址
    dfl.append(df)       #添加df到dfs列表#X_test,
    train_df = pd.concat(dfl, ignore_index=True)  # 将dfs中的数据进行拼接成dataframe存储到train_df
N_CLASSES = np.unique(train_df['ClassId']).size  # 通过统计ClassId不同值的个数，得到交通标志分类数量

print("Number of training images : {:>5}".format(train_df.shape[0]))  # 训练集图片数量
print("Number of classes         : {:>5}".format(N_CLASSES))  # 训练集交通标志分类数量


test_file=glob.glob(os.path.join(Test_IMAGE_DIR, '*/GT-*.csv'))
folder = test_file[0].split('\\')[6]
test_df = pd.read_csv(test_file[0], sep=';')                     #读取文件夹中csv文件
test_df['Filename'] = test_df['Filename'].apply(lambda x: os.path.join(Test_IMAGE_DIR,folder, x))#将‘Filename'列内容延展，增加文件地址
print("Number of test images: {:>5}".format(test_df.shape[0]))
#X_test = resize_image(test_df['Filename'].values) #读取图片并调整尺寸
y_test = np.unique(test_df['ClassId']).size              #读取分类label数据

print(test_df.head())


