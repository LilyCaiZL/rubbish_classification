#coding:utf-8
from flask import request, Flask
import base64
import cv2
import numpy as np
from paddle import fluid
import model1
import test
app = Flask(__name__)
@app.route("/", methods=['POST','GET'])
def classify():
    #将接受的base64代码转化成图片，保存在相对路径，命名为01.png
    img = base64.b64decode(str(request.form['image']))
    image_data = np.fromstring(img, np.uint8)
    image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    cv2.imwrite('01.png', image_data)
    print(image_data)
    _,tensor_img=test.read_image("01.png")
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
    # 初始化推测列表
    result = exe.run(test_program, feed={'image': tensor_img}, fetch_list=[model])
    lab = np.argsort(result)[0][0][-1]
    print(lab)
    lab = str(lab)
    return lab
if __name__ == "__main__":
    app.run("", port=)
