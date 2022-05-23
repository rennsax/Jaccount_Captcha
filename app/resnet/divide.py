import cv2
import numpy as np

def _division(start, img, part):
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

def divide(img) -> list:
    state = 0
    divided_imgs = list()
    for _ in range(5):  # 验证码至多5个字符
        part = np.zeros([200, 200], dtype='uint8')  # 480为高，300为宽，生成黑底图，即默认值为0
        part[:, :] = 255
        state = _division(state, img, part)
        if state != 0:
            divided_imgs.append(part)
    return divided_imgs
