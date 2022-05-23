# Jaccount 验证码识别

说明：该项目是课程 **英特尔前沿AI算法与实践** 的大作业。

该项目中包含：

+ 一篇 ResNet 网络的原论文，来自何晓明博士
+ 用以获取训练集、搭建及训练模型的构建代码
+ 可视化用户端实现（位于 app 文件夹中）

## 项目使用说明

**注：最要不要擅自更改文件夹的名称和位置！**

1. 安装依赖库。

   ```bash
   pip install -r requirements.txt
   ```

2. 如果不想体验从头搭建一个模型，由于模型过大未上传 github，前往链接[交大云盘](https://jbox.sjtu.edu.cn/l/a1SEYL)下载后存入 app/ 文件夹，然后跳至 11；否则请跳至 3。

3. 运行 data_get.py 爬取若干张原始验证码图片，将创建 original_pic 文件夹并自动存入。

4. 运行 noise_1.py 对原始图片进行二值化处理，将创建 pic/close 文件夹并自动存入。

5. 运行 division.py 对二值化后的图片进行分割操作，形成单个的字符，分割结果将存入 divided 文件夹。

6. 运行 noise_2.py 对 divided 文件夹中的图片进行平滑处理，结果存入 smoothed 文件夹。

7. 运行 recognize.py，调用 Google 开源的 *tesseract* 进行字符识别，识别结果将存入 result 文件夹。

8. 识别结果必然不是完美的，未能识别出的图片位于 smoothed/unrecognized 文件夹中，有两种情况：

   + 纯粹没识别出来
   + 分割错误导致未能识别

   除此之外，result 文件夹中的识别结果也会存在错误。需要人工对错误的字符进行重识别，放入正确的文件夹中。

9. 运行 transform.py，将 result 中的识别结果转化为一张 pic.csv 数据集。图片数较多的时候，pic.csv 可能会很大，1000 张图片对应的大小大概是 700MB。

10. 运行 resnet.py，读取 pic.csv 数据集，创建训练集、测试集、验证集，创建模型并进行训练。可以对 resnet.py 中的全局变量做出适当调整。训练出的模型将覆盖 app/model.h5。

11. 运行 app/main.py，选择导入验证码图片进行识别。
