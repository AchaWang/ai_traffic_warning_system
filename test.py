import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# video_url = "https://cctv-ss01.thb.gov.tw/T2-157K+150"
# # 建立 VideoCapture 物件
# vid = cv2.VideoCapture(video_url)

# # 檢查影片是否開啟成功
# if not vid.isOpened():
#     print("無法開啟影片")
#     exit(1)
# while 1:
#     return_value, frame = vid.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#cv2.imshow("Video", frame)

# # 釋放資源
# vid.release()
# cv2.destroyAllWindows()
import cv2

# 指定影像 URL
video_url = "https://cctv-ss01.thb.gov.tw/T2-157K+150"

# 建立 VideoCapture 物件
vid = cv2.VideoCapture(video_url)

# 檢查影片是否開啟成功
if not vid.isOpened():
    print("無法開啟影片")
    exit(1)

# 從影片中讀取每一幀影像
while True:
    # 讀取下一幀影像
    success, frame = vid.read()

    # 檢查是否讀取成功
    if not success:
        vid = cv2.VideoCapture(video_url)
        success, frame = vid.read()


    # 將 BGR 色彩空間轉換為多種色彩空間
    frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    frame_lab2bgr = cv2.cvtColor(frame_lab, cv2.COLOR_Lab2BGR)

    # 顯示影像
    # cv2.imshow("Video", frame)
    cv2.imshow("Video Lab2BGR", frame_lab2bgr)

    # 等待使用者輸入
    key = cv2.waitKey(1) & 0xFF

    # 根據輸入鍵結束程式
    if key == ord('q'):
        break

# 釋放資源
vid.release()
cv2.destroyAllWindows()