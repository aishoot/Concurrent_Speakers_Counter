#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : Created on 2019/2/28 8:04 PM
# @Author  : Chao Peng, EECS, Peking University
# @Purpose : 将所有wav文件提取幅度谱后保存为.mat文件

import os
import librosa
from tqdm import tqdm
import scipy.io as sio
import numpy as np

# 读取数据列表
DATA_PATH = "/mnt/hd8t/pchao/LibriCount10-0dB/test/"   # 服务器下所有数据，语音和json文件一共11440条，单独的5720条
wavfiles = [wav for wav in os.listdir(DATA_PATH) if wav[-3:] == "wav"]   # 所有.wav文件，格式为list

# 读取音频，并确保数据第一维大于等于500
error_labels = {}
all_data = {}
for wav in tqdm(wavfiles):  # 包含地址
    data, _ = librosa.load(DATA_PATH + wav, sr=16000)  # 采样率是16000Hz
    wav_stft = np.abs(librosa.stft(data, n_fft=400, hop_length=160)).T   # (501, 201), 501是时长，201是频点数
    if wav_stft.shape[0] < 500:
        error_labels[wav] = wav_stft.shape
    all_data[wav] = wav_stft

# 保存最后的npy文件
if len(error_labels) == 0:
    print("不存在时长小于500的音频文件.")
else:
    print("时长大于500的音频文件有以下:", error_labels)

sio.savemat('wav_stft.mat', all_data)  # 读取时数据还包含'__version__'等3个无关信息