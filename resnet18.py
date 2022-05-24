# 建模及训练
# by Qiao Guanyuan and Ren Bojun

import numpy as np
import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, Model, Sequential
import tensorflow as tf
tf.test.is_gpu_available()

rate_train_test = 5
norm_size = 224
epochs = 5
dic_class = {chr(i + 97): i for i in range(26)}
class_num = 26
batch_size = 32

raw_data = pd.read_csv("./pic.csv")
row, column = raw_data.shape

row_train = int(row *rate_train_test / (1 + rate_train_test))
raw_data_train = raw_data[:row_train]
raw_data_test = raw_data[row_train:]
raw_data_test.reset_index(drop=True, inplace=True)

x = np.array(raw_data_train.iloc[:, 1:])
y = pd.get_dummies(np.array(raw_data_train.iloc[:, 0]))
X_train, X_validate, y_train, y_validate = train_test_split(x, y, test_size=0.2, random_state=0)

im_rows, im_cols = norm_size, norm_size
input_shape = (im_rows, im_cols, 1)

X_test = np.array(raw_data_test.iloc[:, 1:])
y_test = pd.get_dummies(np.array(raw_data_test.iloc[:, 0]))
img = X_test[random.randint(0, len(X_test) - 1)].reshape(im_rows, im_cols) # 随便挑的一张

X_train = X_train.reshape(X_train.shape[0], im_rows, im_cols, 1)
X_validate = X_validate.reshape(X_validate.shape[0], im_rows, im_cols, 1)
X_test = X_test.reshape(X_test.shape[0], im_rows, im_cols, 1)

# normalisation
X_train = X_train/255
X_validate = X_validate/255
X_test = X_test/255


# 18层
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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


def _make_layer(block, channel, block_num, name, strides=1):
    downsample = None
    if strides != 1:
        downsample = Sequential([
            layers.Conv2D(channel, kernel_size=1, strides=strides,
                          use_bias=False, name=name + "conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name=name + "BatchNorm")
        ], name=name + "shortcut")

    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name=name + "unit_1"))

    for index in range(1, block_num):
        layers_list.append(block(channel, name=name + "unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)


def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=True):
    # tensorflow中的tensor通道排序是NHWC
    # (None, 224, 224, 3)
    # 大小缩小为112*112
    input_image = layers.Input(shape=(im_height, im_width, 1), dtype="float32")
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, name="conv1")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    # 池化层
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    x = _make_layer(block, 64, blocks_num[0], name="block1")(x)
    x = _make_layer(block, 128, blocks_num[1], strides=2, name="block2")(x)
    x = _make_layer(block, 256, blocks_num[2], strides=2, name="block3")(x)
    x = _make_layer(block, 512, blocks_num[3], strides=2, name="block4")(x)

    if include_top:
        # 全连接层
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x

    model = Model(inputs=input_image, outputs=predict)

    return model


def resnet18(im_width=224, im_height=224, num_classes=26, include_top=True):
    return _resnet(BasicBlock, [2, 2, 2, 2], im_width, im_height, num_classes, include_top)


model = Sequential(resnet18())

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

tracker = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_validate, y_validate),
                    verbose=1)


# 保存模型
if not os.path.exists("app"):
    os.mkdir("app")
model.save("./resnet18.h5")

# 读取模型
# model = tf.keras.models.load_model("./app/resnet18.h5",\
#                     custom_objects={"BasicBlock": BasicBlock})

tracker.history['accuracy']

y_pred = model.predict(X_test)


Y_pred_classes = np.argmax(y_pred,axis = 1).reshape(711,1)
#print(Y_pred_classes)

Y_test_classes = np.argmax(np.array(y_test),axis = 1).reshape(711,1)
#print(Y_test_classes)

confusion_matrix(Y_test_classes, Y_pred_classes)

score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

Y_pred_classes = np.argmax(y_pred,axis = 1).reshape(711,1)
#print(Y_pred_classes)

Y_test_classes = np.argmax(np.array(y_test),axis = 1).reshape(711,1)
#print(Y_test_classes)

confusion_matrix(Y_test_classes, Y_pred_classes)

score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])