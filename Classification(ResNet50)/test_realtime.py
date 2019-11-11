import base64
import time

import cv2
import numpy
import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request

import reader
import model
import socket


target_size = [3, 224, 224]
# 定义输入层
image = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
mod = model.ResNet50()
model = mod.net(image, 4)
#模型位置
path = "best_model"
#定义执行器
place=fluid.CUDAPlace(0)
exe=fluid.Executor(place)
#初始化参数
exe.run(fluid.default_startup_program())
[inference_program, feed_target_names, fetch_targets] =fluid.io.load_inference_model(dirname=path, executor=exe)
# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)
#构造InferReader
infer_reader = paddle.batch(reader.test(), 1)
#初始化推测列表
label_list = []
feeder = fluid.DataFeeder(place=place, feed_list=[image])
def cv2_to_PIL(image):
    # image=cv2.imread('lena.png')
    image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    # image.show()
    return image

def PIL_to_cv2(image):
    # image=Image.open('lena.png')
    image=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

    # cv2.imshow('lena',image)
    # cv2.waitKey()
    return image
def draw_bounding_box_on_image(image_path,label_list):
    image = image_path.copy()
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), book[label_list], (255, 255, 0))
    return image
def resize_img(img, target_size):
    img = img.resize(target_size[1:], Image.ANTIALIAS)
    return img
def read_image(img_path):

    origin=Image.open(img_path)
    #origin = img_path.copy()
    img = resize_img(origin, target_size)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    #img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW

    img = np.array(img).astype('float32').transpose((2, 0, 1))
    #img = normalize_image(img)

    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]

    return origin,img

#img = read_image("E:\\深度学习比赛\\rubc\\image_classify_resnet50\\training_dataset\\0\\glass9.jpg")

def make_photo():
    """使用opencv拍照"""
    cap = cv2.VideoCapture(1)  # 默认的摄像头
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("capture", frame)  # 弹窗口
            # 等待按键q操作关闭摄像头
            if cv2.waitKey(1) & 0xFF == ord('q'):
                file_name = "xieyang.jpeg"
                cv2.imwrite(file_name, frame)
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    return file_name

book = {0:"plastic",
        1:"metal",
        2:"glass",
        3:"paper"}
'''
cap = cv2.VideoCapture(1)

cap.set(3,480)
cap.set(3,640)
while cv2.waitKey(1) < 0:
    hasimg, img = cap.read()q

    if not hasimg:
        cv2.waitKey()
        break
'''

cpimage = np.array([0])
while 1:
    _, tensor_img = read_image("01.png")

    if tensor_img.all() == cpimage.all():
        continue
    else:
        result = exe.run(test_program,feed={'image':tensor_img},fetch_list=[model])
        lab = np.argsort(result)[0][0][-1]
        print(book[lab])

        HOST = '192.168.1.104'
        PORT = 7669
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((HOST, PORT))
        sock.listen(5)

        connection, address = sock.accept()
        connection.send(b'%d'%lab)
        time.sleep(3)
        print('Connection success!')
        connection.close()

        cpimage = tensor_img
