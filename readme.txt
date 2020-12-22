请详细查看项目部署文档

数据集（GTSRB）地址https://blog.csdn.net/li_xiaolaji/article/details/108369873


运行环境：
win10操作系统
python3.7
tensorflow 1.14
numpy1.16.4
opencv 4.4


工程文件：
data：存放数据集；
模型的文件夹（效果相对理想）；
model_genggaijihuo：进行了一些对比试验后训练模型的文件夹（效果不理想）；
new_image：用来可视化检测模型效果的图片文件夹；
window_material:存放窗口所需的音频和背景；
signnames.csv：存放每一类标志序号和与其对应名称的文件。


ceshi_won.py：一个测试代码可行性的py文件；
Read_image.py:读取data文件夹中的图片的py文件；
Display_image.py：展示Read_image.py读取到的图片；
image_chan.py：扩充数据集的py文件；
Traffic_Model.py：建立lenet模型的py文件；
Traffic_Train.py：训练模型的py文件；
Traffic_Test.py：加载模型测试模型效果的py文件；
test.py：从new_image中加载一张图片调用模型进行与预测；
测试loss.py测试loss函数的py文件；
window_show.py:使用pyqt所做的一个界面的py文件。
