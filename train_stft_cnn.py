#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : Created on 2019/3/10 9:11 PM
# @Purpose : 使用CNN网络结构训练并测试所有音频数据

import keras
from keras.utils import to_categorical, plot_model
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras import Input, layers, Model
from keras import backend as K

import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # 画图服务器不能使用GUI否则报错
import librosa
import scipy.io as sio
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置gpu编号


# 加载数据
all_data = sio.loadmat('data/stft.mat')

# 读取所有数据和标签, 5720个样本
wavfiles = []
labels = []
for wavfile in tqdm(all_data):  # wavfile是字典中音频的名字
    if wavfile[-3:] != "wav":
        continue
    wavfiles.append(all_data[wavfile])
    label = wavfile[:wavfile.find('_')]  # 字符, '0','1','2',...,'10'
    labels.append(int(label))
wavfiles = np.array(wavfiles)
labels = np.array(labels)

# 将所有labels转化为one-hot向量
y_hot = to_categorical(labels)

# 划分训练集和测试集
X_train, X_test, y_train_hot, y_test_hot = train_test_split(wavfiles, y_hot, test_size=0.2, random_state=42)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train_hot:", y_train_hot.shape)
print("y_test_hot:", y_test_hot.shape)

# 相关参数设置
lr = 0.001
batch_size = 128
#drop_out_rate = 0.5
input_shape = (X_train.shape[1], X_train.shape[2])  # (501, 201)
num_classes = 11   # [0-10]一共11类
epoch = 100

# 搭建卷积神经网络，Conv1D Model
input_tensor = Input(shape=(input_shape))
x = layers.Conv1D(64, 3, padding='valid', activation='relu', strides=1)(input_tensor)
x = layers.Conv1D(32, 3, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(3)(x)
x = layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1)(x)
x = layers.Conv1D(64, 3, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(3)(x)
x = layers.Dropout(0.25)(x)

x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.1)(x)

x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
output_tensor = layers.Dense(num_classes, activation='softmax')(x)

model = Model(input_tensor, output_tensor)
model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(lr = lr),
             metrics=['accuracy'])
print(model.summary())

# 开始训练并保存模型，validation_split=0.25
history = model.fit(X_train, y_train_hot, validation_data=[X_test, y_test_hot],
          batch_size=batch_size, epochs=epoch)
model.save("models/stft_cnn_%sepoch.h5"%(epoch))

# 评价在所有数据集上的得分
score = model.evaluate(wavfiles, y_hot, batch_size=batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 保存训练和验证过程
#plot_model(model, to_file='model.png', show_shapes=True)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim((0, 1.0))  # 纵坐标
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("pictures/model_accuracy.png")
#plt.show()

# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("model_loss.png")
#plt.show()
