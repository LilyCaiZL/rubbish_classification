import paddle as paddle
import paddle.fluid as fluid
import reader
import model
import itertools
import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    这个函数用于绘制混淆矩阵；
    可以通过设置normalize=True应用均一化；
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
label_names = [str(i) for i in range(4)]
label_class = {name: index for index, name in enumerate(label_names)}

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
# 定义一个使用GPU的执行器
place = fluid.CUDAPlace(0)
#place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
#定义fetch_list
fetch_list = [ image , label , model ,  avg_cost.name , acc.name ]

val_reader = paddle.batch(reader.train(), batch_size=64)
# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
#调用SKlearn包中的混淆矩阵方法
from sklearn.metrics import confusion_matrix
path = "best_model"
[inference_program, feed_target_names, fetch_targets] =fluid.io.load_inference_model(dirname=path, executor=exe)
test_accs=[]
test_costs=[]
c_matrix=[]
c_matrix=np.array(c_matrix)
for batch_id,data in enumerate(val_reader()):
	doc=exe.run(test_program,feeder.feed(data),fetch_list)
	test_accs.append(doc[4])
	test_costs.append(doc[3])
	pred=np.argsort(doc[2])[:,-1]
	pred=pred.reshape(pred.shape[0],1)
	c_matrix=confusion_matrix(doc[1],pred)
#求测试Acc，loss平均值
test_cost=(sum(test_costs)/len(test_costs))
test_acc=(sum(test_accs)/len(test_accs))
plt.figure()
plot_confusion_matrix(c_matrix,classes=label_names, normalize=True,title='Normalized confusion matrix')
print('测试结果：Cost:%05.2f,Accuracy:%5.2f'%(test_cost,test_acc))

