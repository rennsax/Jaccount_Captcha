# 对原始图片进行二值化和除噪处理
# by Xie Tong

import cv2
import os
import numpy as np
import glob
INPUT_PATH = './original_pic'
# OUTPUT_PATH_1 = './pic/binary'
# OUTPUT_PATH_2 = './pic/erode'
OUTPUT_PATH_3 = './pic/close'
# OUTPUT_PATH_4 = './pic/dilate'
# OUTPUT_PATH_5 = './pic/open'
# IMG_NAME = '/green'
# if not os.path.exists(OUTPUT_PATH_1): os.makedirs(OUTPUT_PATH_1)
# if not os.path.exists(OUTPUT_PATH_2): os.makedirs(OUTPUT_PATH_2)
if not os.path.exists(OUTPUT_PATH_3): os.makedirs(OUTPUT_PATH_3)
# if not os.path.exists(OUTPUT_PATH_4): os.makedirs(OUTPUT_PATH_4)
# if not os.path.exists(OUTPUT_PATH_5): os.makedirs(OUTPUT_PATH_5)

mask_files = glob.glob(INPUT_PATH + '/*.jpg')
# mask_files.sort(key=lambda s: int(s.split('\\')[-1].split('.jpg')[0]))

for i in range(len(mask_files)):
    img = cv2.imread(mask_files[i])

    img = cv2.resize(img, None, fx=5, fy=5,
                  interpolation=cv2.INTER_CUBIC)#调整图片大小

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, binary_img =cv2.threshold(img,200,255,cv2.THRESH_BINARY)

    binary_img1 = cv2.erode(binary_img, np.ones((3, 3), np.uint8))
    binary_img2 = cv2.morphologyEx(binary_img1, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # binary_img3 = cv2.dilate(binary_img2, np.ones((5, 5), np.uint8))
    # result = cv2.morphologyEx(binary_img3, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # cv2.imwrite(f'{OUTPUT_PATH_1}/{i + 1}.png', binary_img)
    # cv2.imwrite(f'{OUTPUT_PATH_2}/{i + 1}.png', binary_img1)
    cv2.imwrite(f'{OUTPUT_PATH_3}/{i + 1}.png', binary_img2)
    # cv2.imwrite(f'{OUTPUT_PATH_4}/{i + 1}.png', binary_img3)
    # cv2.imwrite(f'{OUTPUT_PATH_5}/{i + 1}.png', result)
    print(f'{i + 1}.png')

# cv2.imshow('Binary Image', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()