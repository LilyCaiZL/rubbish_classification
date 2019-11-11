# 基于图像分类和目标识别的可回收垃圾分类机
运用PaddlePaddle 深度学习框架，构架了基
于MobilenetSSD 和ResNet50 深度神经网络
的边云协同模型，进行可回收垃圾的图像分
类和目标识别，图像分类识别准确度可以达
到 88%，目标识别准确度可以达到65%。

## 一、核心技术概括
- 图像分类：基于百度PaddlePaddle 深度学习框架构建综合
ResNet50 和迁移学习的垃圾分类模型。
- 目标识别：基于百度PaddlePaddle 深度学习框架构建
MobileNetSSD 实时可回收垃圾目标识别模型。
- 边缘计算：利用PaddleLite和模型优化压缩工具，将压缩模型部署在树莓派3B+，连接舵机，实现模型应用于硬件。
- 云端计算：利用花生壳和Flask将模型部署在远端主机，树莓派采集图像，主机接受图像返回结果，并连接，实现模型应用于硬件。

## 二、场景运用
- 分别在主机、嵌入式系统、移动设备上实现基于
百度PaddlePaddle 深度学习框架的可回收垃圾
自动分类、识别系统。
- 小区（社区）：智能垃圾桶，进行可回收垃圾的
自动分类。
- 垃圾场（收集站）：自动化流水线，进行可回收
垃圾的目标识别。

## 三、项目最终效果
- 图像分类测试图

![Demo](images/classification_result.jpg)
- 目标识别测试图

![Demo](images/object_result.jpg)
## 四、数据集介绍
- 本设计使用的，数据来源于斯坦福大学的可回收垃圾集（https://github.com/garythung/trashnet） 和自行调查拍摄的可回收垃圾集共同组成的数据集

1、图像分类

| 编号 | 类别 | 数量 |  
|--|--|--|
| 1 | metal | 414 |
| 2 | glass | 508 |
| 3 | paper | 602 |
| 4 | plastic | 518 |

2、目标识别

| 编号 | 类别 | 数量 |  
|--|--|--|
| 1 | metal | 414 |
| 2 | glass | 508 |
| 3 | paper | 602 |
| 4 | plastic | 518 |
## 五、超参数设置
- 图像分类：
迁移学习:
inputsize = 3,224,224
LearningRate = 0.0001
BatchSize = 16

Train:
inputsize = 3,224,224
LearningRate = 0.001
BatchSize = 32

- 目标识别：
inputsize = 3,300,300
LearningRate = 0.001
BatchSize = 32
## 六、训练及测试项目运行方式
### 1、模型训练与测试
- 配置要求

训练使用百度提供的AI Studio

| 编号 | 项目 | 配置 |  
|--|--|--|
| 1 | GPU | v100 |
| 2 | 显存 |  16GB|
| 3 | CPU | 8 |
| 4 | RAM | 32G |
| 5 | python | 3.5 |
a.ResNet50图像分类
- 训练

进入Classification(ResNet50)目录
```
python3 train.py
```
- 测试

```
python3 test_realtime.py
```
b.MobileNetSSD目标识别
进入Object Detection(SSD)目录
- 训练

```
python3 train.py
```
- 测试

```
python3 _ce.py
```
### 2、边缘计算（目前仅实现目标识别）
- 配置要求

| 编号 | 项目 | 配置 |  
|--|--|--|
| 1 | 树莓派 | 3B+ |
| 2 | 树莓派摄像头 |  官方摄像头|
| 3 | 操作系统 | 官方操作系统 |
| 4 | 舵机 | SP90 |
| 5 | cmake | 3.10 |
- 运行
关机状态将摄像头连上树莓派
将舵机通过杜邦线连树莓派引脚
将PaddleLite-armlinux-demo目录拷贝进树莓派
进入object_detection_demo目录


```
./r.sh
```
运行即可
### 3.伪云端（目前仅实现图像分类）
## 七、详细方案
![Demo](images/xiangxi.jpg)
### 1、模型训练与测试

### 2、边缘端部署与使用

### 3、云端部署与使用
