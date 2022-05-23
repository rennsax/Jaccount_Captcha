# 将图片数据集转化为 csv 表格存储
# by Ren Bojun

import cv2
import numpy as np
import os
import random
import csv
import numpy as np
# import tensorflow as tf
# tf.test.is_gpu_available()

# 全局参数
norm_size = 224
data_path = "./classify/"
dic_class = {chr(i + 97): i for i in range(26)}

files = os.listdir(data_path)
folders = list()
for folder in files:
    folders.append(os.path.join(data_path, folder))

data_rows = list()

# 将图片进行缩放，并存入 data_row 列表
for folder in folders:
    label = dic_class[folder[-1]]
    img_list = os.listdir(folder)
    for img in img_list:
        img_path = os.path.join(folder, img)
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (norm_size, norm_size), cv2.INTER_LINEAR)
        img_ndarray = np.array(img).flatten()
        img_ndarray = np.insert(img_ndarray, 0, label)
        data_rows.append(img_ndarray)

# 打乱数据集 （其实没有必要）
random.shuffle(data_rows)

# 存入 csv
data_set_title = ["label"]
for i in range(norm_size*norm_size):
    data_set_title.append("pixel%d"%i)

with open("./pic.csv", 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=data_set_title)
    w.writeheader()
    for data in data_rows:
        i = 0
        tmp = {}
        for str in data_set_title:
            tmp[str] = data[i]
            i += 1
        w.writerow(tmp)