# 使用 Google 开源的 OCR 模型 tesseract 进行初次分类
# by Ren Bojun

import os
import shutil
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract


def organize():

    dir_list = os.listdir(path)
    if dir_name not in dir_list or not os.path.isdir(path + dir_name):
        os.mkdir(path + dir_name)
        print("未找到合适目录，已自动创建。")
    else:
        dir_list.remove(dir_name)
        print("%s目录已经存在。"%dir_name)

    target = path + dir_name

    for dir in dir_list:
        if not os.path.isdir(path + dir):
            continue
        pic_list = os.listdir(path + dir)
        i = 0
        for pic in pic_list:
            os.rename(path + dir + '/' + pic, target + "/%s_%d.jpg"%(dir, i))
            i += 1
        os.removedirs(path + dir)

if __name__ == '__main__':
    path = r"./smoothed/"
    dir_name = r"unrecognized"
    # organize()
    # shutil.rmtree("./result")

    # "./result/" 目录用于存储初次识别后的字符图片，需要人工检查后移至 "./classify" 目录
    target_path = r'./result/'

    if not os.path.exists(target_path):
        os.mkdir(target_path)
    # 创建 a-z 文件夹
    for i in range(97, 123):
        if not os.path.exists(target_path + '%s/'%chr(i)):
            os.mkdir(target_path + '%s/'%chr(i))

    pic_list = os.listdir(path + dir_name)
    for pic in pic_list:
        img = Image.open(path + dir_name + '/' + pic)
        cong = r'--psm 10 --oem 3 -l eng'
        ch = pytesseract.image_to_string(img, config=cong).strip()
        if ch == "":
            # 未识别出
            print(pic, "识别失败！")
            continue
        ch = ch[0]
        if 65 <= ord(ch) <= 90:
            ch = chr(ord(ch) + 32)
        # l 有很大概率被识别为 |，加入更正
        elif ch == '|':
            ch = 'l'
        elif not 97 <= ord(ch) <= 122:
            print(pic, "识别失败！")
            continue
        shutil.move(path + dir_name + '/' + pic, target_path + ch)
        print(pic, "识别成功，为 %s"%ch)

"""
Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR.
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
                        bypassing hacks that are Tesseract-specific.

OCR Engine modes:

 0     Legacy engine only.
 1     Neural nets LSTM engine only.
 2     Legacy + LSTM engines.
 3     Default, based on what is available.
"""