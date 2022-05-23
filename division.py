# 分割算法：将二值化后的图片分割为单个字符
# by Qiao Guanyuan

import cv2
import numpy as np
import os

part = np.zeros([200, 200], dtype='uint8')  # 480为高，300为宽，生成黑底图，即默认值为0
part[:, :] = 255


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


# 边缘检测算法分割
def division(start, img):
    height = img.shape[0]
    width = img.shape[1]
    a, b, c, d = 0, 0, 0, 0

    for j in range(start, width):
        a = 0
        for i in range(height):
            if img[i][j] != 0:
                continue
            else:
                a = j  # 获取左边界
                break
        if a != 0:
            break
    # 如果没有左边界，即分割完毕，返回0
    if a == 0:
        return 0

    for j in range(a + 1, width):
        k = 0
        # if j - a > 85:
        #     b = j - 1
        #     break#本为应对粘连字符，因字符宽度不一致已废弃
        for i in range(0, height):
            if img[i][j] == 255:
                k += 1
        if k == height:
            b = j - 1  # 获取右边界
            break

    for i in range(height):
        c = 0
        for j in range(a, b + 1):
            if img[i][j] != 0:
                continue
            else:
                c = i  # 获取上边界
                break
        if c != 0:
            break

    for i in range(c, height):
        k = 0
        for j in range(a, b + 1):
            if img[i][j] == 255:
                k += 1
        if k == b + 1 - a:
            d = i - 1  # 获取下边界
            if d - c < 50:
                continue
            else:
                break
    first = img[c:d + 1, a:b + 1]  # 裁切
    x = (200 - (b + 1 - a)) // 2
    y = (200 - (d + 1 - c)) // 2
    part[y:y + (d + 1 - c), x:x + (b + 1 - a)] = first

    return b + 1  # 返回右边界作为下一次分割起点


def main():
    for i in range(1, int(max(os.listdir('./original_pic/'), key = lambda s: int(s[:-4]))[:-4]) + 1):
        file = f"divided/{i}"
        mkdir(file)  # 为每张图片创建文件夹

    for l in range(1, int(max(os.listdir('./original_pic/'), key = lambda s: int(s[:-4]))[:-4]) + 1):
        print(l)
        img = cv2.imread(f"./pic/close/{l}.png", 0)
        state = 0
        for time in range(5):  # 验证码至多5个字符
            state = division(state, img=img)
            if state == 0:
                part[:, :] = 255  # 重置底图
                break
            else:
                cv2.imwrite(f"divided/{l}/part{time}.jpg", part)
                part[:, :] = 255


if __name__ == '__main__':
    main()
