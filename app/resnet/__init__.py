import os
import cv2
import numpy as np
from . import noise, divide
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

def load_model(model_path, log = True):
    if not log:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    class BasicBlock(layers.Layer):
        def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
            super(BasicBlock, self).__init__(**kwargs)
            self.out_channel = out_channel
            self.strides = strides
            self.downsample = downsample

            self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                    padding="SAME", use_bias=False)
            # Batch Normalization,加快模型训练时的收敛速度，使得模型训练过程更加稳定，
            # 避免梯度爆炸或者梯度消失。并且起到一定的正则化作用
            self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

            self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                    padding="SAME", use_bias=False)
            self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

            self.downsample = downsample
            self.relu = layers.ReLU()
            self.add = layers.Add()

        def call(self, inputs, training=False):
            identity = inputs
            # 为了区别两种结构
            if self.downsample is not None:
                identity = self.downsample(inputs)

            x = self.conv1(inputs)
            x = self.bn1(x, training=training)
            x = self.relu(x)

            x = self.conv2(x)
            x = self.bn2(x, training=training)

            x = self.add([identity, x])
            x = self.relu(x)

            return x

        def get_config(self):

            config = super().get_config().copy()
            config.update({
                'out_channel': self.out_channel,
                'strides': self.strides,
                'downsample': self.downsample,
            })
            return config

    model = tf.keras.models.load_model(model_path,\
                    custom_objects={"BasicBlock": BasicBlock})
    return model


def recognize(img_path: str, model_path: str = "./resnet18.h5", log = True, model = None) -> str:
    if not os.path.exists(img_path):
        raise FileNotFoundError("文件%s不存在！"%img_path)
    ### 可以做一个判断是否为图片
    img = cv2.imread(img_path)
    img = noise.close(img)
    parts = divide.divide(img)
    noise.cut_noise(parts)
    if model is None:
        model = load_model(model_path, log=log)
    size = len(parts)
    X = np.ones((size, 224, 224))
    for i in range(size):
        matrix = parts[i]
        matrix = matrix/255
        X[i] = matrix
    Y = model.predict(X)
    Y_classes = np.argmax(Y,axis = 1).reshape(size,1)
    result = ""
    dic_class = {i: chr(i + 97) for i in range(26)}
    for y in Y_classes:
        result += dic_class[y[0]]
    return result