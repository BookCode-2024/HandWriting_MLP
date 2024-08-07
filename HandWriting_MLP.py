# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:32:32 2021

@author: zhaodf
"""
#采用MLP模型对MNIST手写数字进行识别
#MNIST是一个手写体数字的图片数据集，该数据集来由美国国家标准与技术研究所
# （National Institute of Standards and Technology (NIST)）发起整理，
#一共统计了来自250个不同的人手写数字图片，其中50%是高中生，50%来自人口普查局的工作人员。
#在上述数据集中，训练集一共包含了 60,000 张图像和标签，而测试集一共包含了 10,000 张图像和标签。

from keras.datasets import mnist
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()#加载数据

plt.subplot(241)
plt.imshow(X_train[12], cmap=plt.get_cmap('gray'))
plt.subplot(242)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(243)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(244)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.subplot(245)
plt.imshow(X_train[4], cmap=plt.get_cmap('gray'))
plt.subplot(246)
plt.imshow(X_train[5], cmap=plt.get_cmap('gray'))
plt.subplot(247)
plt.imshow(X_train[6], cmap=plt.get_cmap('gray'))
plt.subplot(248)
plt.imshow(X_train[7], cmap=plt.get_cmap('gray'))

plt.show()

from keras.models import Sequential # 导入Sequential模型
from keras.layers import Dense # 全连接层用Dense类
from keras.utils import np_utils # 导入np_utils是为了用one hot encoding方法将输出标签的向量（vector）转化为只在出现对应标签的那一列为1，其余为0的布尔矩阵

(X_train,y_train),(X_test,y_test) = mnist.load_data() #加载数据
#print(X_train.shape[0])
#数据集是3维的向量（instance length,width,height).对于多层感知机，模型的输入是二维的向量，因此这里需要将数据集reshape，即将28*28的向量转成784长度的数组。可以用numpy的reshape函数轻松实现这个过程。
num_pixels = X_train.shape[1] * X_train.shape[2] 
X_train = X_train.reshape(X_train.shape[0],num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0],num_pixels).astype('float32')

#给定的像素的灰度值在0-255，为了使模型的训练效果更好，通常将数值归一化映射到0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# 搭建神经网络模型，创建一个函数，建立含有一个隐层的神经网络
def baseline_model():
    model = Sequential() # 建立一个Sequential模型,然后一层一层加入神经元
    # 第一步是确定输入层的数目：在创建模型时用input_dim参数确定。例如，有784个个输入变量，就设成num_pixels。
    #全连接层用Dense类定义：第一个参数是本层神经元个数，然后是初始化方式和激活函数。这里的初始化方法是0到0.05的连续型均匀分布（uniform），Keras的默认方法也是这个。也可以用高斯分布进行初始化（normal）。
    model.add(Dense(5,input_dim=num_pixels,kernel_initializer='uniform',activation='relu'))
    # model.add(Dense(10,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(num_classes,kernel_initializer='uniform',activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return model

model = baseline_model()
model.summary()
history=model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=100, batch_size=200) #训练

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

#fig = plt.figure()
figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['acc'],'-r',label='training acc',linewidth=1.5)
plt.plot(history.history['val_acc'],'-b',label='val acc',linewidth=1.5)
plt.title('model accuracy',font2)
plt.ylabel('accuracy',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='lower right',prop=font2)

#fig = plt.figure()
figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['loss'],'-r',label='training loss',linewidth=1.5)
plt.plot(history.history['val_loss'],'-b', label='val loss',linewidth=1.5)
plt.title('model loss',font2)
plt.ylabel('loss',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='upper right',prop=font2)

figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['acc'],'-g',label='training acc',linewidth=1.5)
plt.plot(history.history['val_acc'],'-r',label='val acc',linewidth=1.5)
plt.plot(history.history['loss'],'-y',label='training loss',linewidth=1.5)
plt.plot(history.history['val_loss'],'-b', label='val loss',linewidth=1.5)
plt.title('model loss and accuracy',font2)
plt.ylabel('value',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='best',prop=font2)

scores = model.evaluate(X_test,y_test) #model.evaluate 返回计算误差和准确率
print(scores)
print("Base Error:%.2f%%"%(100-scores[1]*100))
