import cv2
import numpy as np

def close(img):
    img = cv2.resize(img, None, fx=5, fy=5,
                  interpolation=cv2.INTER_CUBIC)#调整图片大小

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, binary_img =cv2.threshold(img,200,255,cv2.THRESH_BINARY)

    binary_img1 = cv2.erode(binary_img, np.ones((3, 3), np.uint8))
    binary_img2 = cv2.morphologyEx(binary_img1, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return binary_img2

def cut_noise(img_list: list) -> None:
    for i in range(len(img_list)):
        tmp_img = img_list[i]
        tmp_img = cv2.morphologyEx(tmp_img, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        # 将图片重置为 224*224
        tmp_img = cv2.resize(tmp_img, (224, 224), cv2.INTER_LINEAR)
        img_list[i] = tmp_img