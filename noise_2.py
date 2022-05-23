# 对分割后的单个字符图片进行平滑处理，提高分割精确度
# by Xie Tong

import cv2
import os
import numpy as np
import glob
INPUT_PATH = './divided'
OUTPUT_PATH = './smoothed'
# IMG_NAME = '/green'
if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

folders = os.listdir(INPUT_PATH)
r = int(max(folders, key = lambda s: int(s)))
l = int(min(folders, key = lambda s: int(s)))

for j in range(l, r + 1):
    cur_out_path = OUTPUT_PATH + '/' + str(j)
    if not os.path.exists(cur_out_path): os.makedirs(cur_out_path)
    mask_files = glob.glob(INPUT_PATH + '/' + str(j) + '/*.jpg')
    # mask_files.sort(key=lambda s: int(s.split('\\')[-1].split('.jpg')[0]))

    for i in range(len(mask_files)):
        img = cv2.imread(mask_files[i])

        # img = cv2.resize(img, None, fx=5, fy=5,
        #               interpolation=cv2.INTER_CUBIC)#调整图片大小

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        cv2.imwrite(f'{cur_out_path}/{i + 1}.png', img)
        print(f'{j+1}_{i + 1}.png')

# img = cv2.imread('./933_2.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
# cv2.imshow('Binary Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()