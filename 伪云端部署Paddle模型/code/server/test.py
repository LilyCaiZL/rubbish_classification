import base64
import numpy
import paddle.fluid as fluid
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request
import model1
def resize_img(img, target_size):
    img = img.resize(target_size[1:], Image.ANTIALIAS)
    return img
def read_image(img_path):
    target_size = [3, 224, 224]
    origin=Image.open(img_path)
    #origin = img_path.copy()
    img = resize_img(origin, target_size)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    #img = normalize_image(img)
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]

    return origin,img

book = {0:"plastic",
        1:"metal",
        2:"glass",
        3:"paper"}

if __name__ == '__main__':
    target_size = [3, 224, 224]
    # 定义输入层
    image = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
    mod = model1.ResNet50()
    model = mod.net(image, 4)
    # 模型位置
    path = "best_model"
    # 定义执行器
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    # 初始化参数
    exe.run(fluid.default_startup_program())
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)
    # 获取训练和测试程序
    test_program = fluid.default_main_program().clone(for_test=True)
    # 构造InferReader
    #infer_reader = paddle.batch(reader.test(), 1)
    # 初始化推测列表
    label_list = []
    feeder = fluid.DataFeeder(place=place, feed_list=[image])
    _,tensor_img = read_image("")
    result = exe.run(test_program, feed={'image': tensor_img}, fetch_list=[model])
    lab = np.argsort(result)[0][0][-1]
    print(lab)
