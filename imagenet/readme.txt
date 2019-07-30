main.py：包含多种训练方式（单结点单GPU，多节点单GPU，单结点多GPU，多节点多GPU），适合于pytorch1.0及以上版本。
main2.py：我根据自己服务器的情况做的修改，运行有问题，还在修改。

1、 训练
要训练模型，请运行main.py，指定所需要的模型和ImageNet数据集的路径。
python main.py -a resnet18 Data/CLS-LOC

默认学习率计划从0.1开始，每30个时期衰减10倍。这适用于具有批量标准化的ResNet和模型，但对于AlexNet和VGG来说太高了。使用0.01作为AlexNet或VGG的初始学习率
python main.py -a alexnet --lr 0.01 Data/CLS-LOC

2、 多进程分布式数据并行训练
您应该始终使用NCCL后端进行多处理分布式培训，因为它目前提供最佳的分布式培训性能。
2.1 单节点多GPUs
python main.py -a resnet50 --lr 0.01 --dist-url 'tcp://222.20.79.232:50021' --dist-backend 'nccl' --world-size 1 --rank 0 Data/CLS-LOC
2.2 多节点多GPUs

3、 Usage
python main.py -h
查看所有参数。

Error:
1、AttributeError: module 'torch.multiprocessing' has no attribute 'spawn'
--multiprocessing-distributed 在pytorch0.4.1版本不可用，它是1.0版本中的。在命令中去掉即可。或者升级到1.0版本。

2、RuntimeError: CUDA error: out of memory
显存错误。
原因：batch_size太大
ps aux|grep zhanglanlan|grep python
得到python的所有进程号
然后kill -s 9 进程号

查看运行在gpu上的所有程序
fuser -v /dev/nvidia*

运行程序的命令：
python main.py -a vgg16 -b 128 --lr 0.001 --dist-url 'tcp://222.20.79.232:50021' --dist-backend 'nccl' --world-size 1 --rank 0 Data/CLS-LOC

