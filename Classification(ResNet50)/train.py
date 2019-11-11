import warnings
import paddle as paddle
import paddle.fluid as fluid
import reader
import os
import numpy as np
from sklearn.metrics import *
import model


warnings.filterwarnings('ignore')
# 解压好的 天气图片 路径
files_path = 'E:\\深度学习比赛\\rubc\\image_classify_resnet50\\training_dataset'

label_names = [str(i) for i in range(4)]
label_class = {name: index for index, name in enumerate(label_names)}
# 列出所有数据集里的图片，给出 (路径, 类名) 的列表
# - 训练时，使用每个分类里前 80%
# - 测试时，使用每个分类里后 20%
def list_files(basepath, mode='all'):
	assert mode in ['train', 'test', 'all']

	files = []
	for name in label_names:
		class_files = sorted(os.listdir(os.path.join(basepath, name)))

		part = int(len(class_files) * 0.9)
		if mode == 'train':
			class_files = class_files[:part]
		elif mode == 'test':
			class_files = class_files[part:]

		for file in class_files:
			files.append((os.path.join(basepath, name, file), name))
	return files

# 取得文件列表
train_files = list_files(files_path, mode='train')
test_files = list_files(files_path, mode='test')

np.random.shuffle(train_files)
np.random.shuffle(test_files)

with open('train_list.txt','w') as f:
    for line in train_files:
        f.write(str(line).replace('(','').replace(')','').replace("'",""))
        f.write('\n')
with open('val_list.txt','w') as f:
    for line in test_files:
        f.write(str(line).replace('(','').replace(')','').replace("'",""))
        f.write('\n')

train_reader = paddle.batch(reader.train(), batch_size=64)
val_reader = paddle.batch(reader.train(), batch_size=64)

# 定义输入层
image = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取分类器
mod = model.ResNet50()
model = mod.net(image,4)
confidence = fluid.layers.softmax(model)
# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=confidence, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=confidence, label=label)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)


# 定义一个使用GPU的执行器
place = fluid.CUDAPlace(0)
#place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 经过step-1处理后的的预训练模型
#pretrained_model_path = 'models/step-1_model/'

# 加载经过处理的模型
#fluid.io.load_params(executor=exe, dirname=pretrained_model_path)

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
#f1_score calculate
best_f1_score = 0
# 训练115次
param_path = 'param_model'

test_id = 0
best_f1_score = 0
for pass_id in range(115):
    # 进行训练
    y_true_train = []
    y_pred_train = []
    for batch_id, data in enumerate(train_reader()):
        result, ori_label,train_cost = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[confidence, label, avg_cost])
        y_pred_train.append(np.argsort(result[0])[-1])
        y_true_train.append(ori_label[0][0])

    f1score=f1_score(y_true_train, y_pred_train, average='weighted')
    accuracy = accuracy_score(y_true_train, y_pred_train)
    print('Pass:%d, loss:%0.5f, accuracy:%0.5f, F1_Score:%0.5f' % (pass_id, train_cost[0], accuracy, f1score))
    #保留训练参数模型
    #fluid.io.save_persistables(executor=exe, dirname=param_path,main_program=fluid.default_startup_program())

    if pass_id % 10 == 0:
        fluid.io.save_persistables(executor=exe, dirname=param_path+"\\"+str(pass_id), main_program=fluid.default_startup_program())
        y_true_val = []
        y_pred_val = []
        for batch_id, data in enumerate(val_reader()):
            result, ori_label, train_cost = exe.run(program=fluid.default_main_program(),
                                                    feed=feeder.feed(data),
                                                    fetch_list=[confidence, label, avg_cost])
            y_pred_val.append(np.argsort(result[0])[-1])
            y_true_val.append(ori_label[0][0])
        f1score_val = f1_score(y_true_val, y_pred_val, average='weighted')
        accuracy_val = accuracy_score(y_true_val, y_pred_val)
        print('test:%d, loss:%0.5f, accuracy:%0.5f, F1_Score:%0.5f' % (test_id, train_cost[0], accuracy_val, f1score_val))
        test_id += 1
        if best_f1_score < f1score_val:
            best_f1_score = f1score_val
            fluid.io.save_inference_model("best_model", ['image'], [confidence], exe)
