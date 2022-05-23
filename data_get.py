# 访问 jAccount 登录界面，爬取验证码原始图片
# by Ren Bojun

import os
import requests

url = r"https://jaccount.sjtu.edu.cn/jaccount/captcha"
headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36"}

max_wrong_times = 10
times = 1000

def main():
    res = requests.get(url, headers=headers)
    if (res.status_code != 200):
        wrong_times += 1
        return
    with open("./original_pic/{index}.jpg".format(index = i), 'wb') as f:
        f.write(res.content)
        print("爬取第 {index} 张图片成功！".format(index = i))

wrong_times = 0
i: int
try:
    i = int(max(os.listdir('./original_pic/'), key = lambda s: int(s[:-4]))[:-4]) + 1
except ValueError:
    # 如果没有图片，则从 1 开始标号
    i = 1

if __name__ == "__main__":
    print("从第 %d 张图片开始爬取，按任意键确认。"%i)
    if input() != 'q':
        while wrong_times <= max_wrong_times and times > 0:
            main()
            i += 1
            times -= 1
