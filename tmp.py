import cv2 
import json 
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


# model = YOLO("/home/zhangqin/wangjl_data/meter_reading_recognition/weights/seg_scale_with_green.pt")
# res = model("/home/zhangqin/wangjl_data/meter_reading_recognition/test_dataset/0_nrxt_sub_defect_dir2_51_230504_seg_biaoji.jpg")[0]
# print(res)
import cv2
def getpic(videoPath, svPath):#两个参数，视频源地址和图片保存地址
    cap = cv2.VideoCapture(videoPath)

    numFrame = 0
    while True:
        # 函数cv2.VideoCapture.grab()用来指向下一帧，其语法格式为：
        # 如果该函数成功指向下一帧，则返回值retval为True
        if cap.grab():
            # 函数cv2.VideoCapture.retrieve()用来解码，并返回函数cv2.VideoCapture.grab()捕获的视频帧。该函数的语法格式为：
            # retval, image = cv2.VideoCapture.retrieve()image为返回的视频帧，如果未成功，则返回一个空图像。retval为布尔类型，若未成功，返回False；否则返回True
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                numFrame += 1
                #设置图片存储路径
                newPath = svPath + str(numFrame).zfill(5) + ".png"
                #注意这块利用.zfill函数是为了填充图片名字，即如果不加这个，那么图片排序后大概就是1.png,10.png,...2.png,这种顺序合成视频就会造成闪烁，因此增加zfill，变为00001.png,00002.png;可以根据生成图片的数量大致调整zfill的值
                # cv2.imencode()函数是将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输。
                cv2.imencode('.png', frame)[1].tofile(newPath)
                print(numFrame)
        else:
            break

num = 4
getpic(f"xiada/{num}.mp4", f"xiada/{num}/")
print("all is ok")
