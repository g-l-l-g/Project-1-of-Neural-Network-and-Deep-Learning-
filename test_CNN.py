import os
import json
import sys
import cupy as cp
sys.path.append(r"D:\python object\neural network\project1\codes_gpu\mynn")
sys.path.append(r"D:\python object\neural network\project1\codes_gpu\mynn\op.py")
from mynn import models, optimizer, metric, lr_scheduler, runner, download_dataset, op, early_stopping
from mynn.activation_function import ReLU, Tanh, Logistic
from op import Linear, Conv2D, MaxPooling2D, Flatten


def get_ith(FILE_NAME):
    try:
        with open(FILE_NAME, "r") as f:
            ith = int(f.read().strip())
    except FileNotFoundError:
        ith = 0
    ith += 1
    with open(FILE_NAME, "w") as f:
        f.write(str(ith))
    return ith


def js_save(file_n, data_, save_dir):
    file_path = os.path.join(save_dir, file_n)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data_, file, indent=4)


def cnn_train(
        layers_=(),
        save_dir=None,
        dataset_dir=None,
        data_size=None,
        Loss_function=None,
        Optimizer=None,
        Scheduler=None,
        Runner=None,
        Early_Stopping=None,
        ) -> cp.ndarray:

    # 数据加载与预处理
    X_train, y_train, X_val, y_val, X_test, y_test = download_dataset.load_dataset(dataset_dir)

    # 数据规格转换为(N, C, H, W), 具体见
    channels = data_size['channels']
    height = data_size['height']
    width = data_size['width']
    X_train = download_dataset.reshape_images(X_train, channels, height, width)
    X_val = download_dataset.reshape_images(X_val, channels, height, width)
    X_test = download_dataset.reshape_images(X_test, channels, height, width)

    # 模型配置
    layers_ = list(layers_) if isinstance(layers_, tuple) else layers_
    model = models.CNN(layers=layers_)

    # 训练组件配置
    # 损失函数配置
    if Loss_function['method'] == 'MultiCrossEntropyLoss':
        loss_function_ = op.MultiCrossEntropyLoss(model=model)
    else:
        raise ValueError('loss_function must be "MultiCrossEntropyLoss"')

    # 优化器配置
    if Optimizer['method'] == 'SGD':
        optimizer_ = optimizer.SGD(init_lr=Optimizer["init_lr"], model=model)
    elif Optimizer['method'] == 'MomentumGD':
        optimizer_ = optimizer.MomentumGD(init_lr=Optimizer["init_lr"], init_beta=Optimizer['init_beta'], model=model)
    else:
        raise ValueError('optimizer must be "SGD" or "MomentumGD"')

    # 学习率下降配置
    if Scheduler['method'] == 'StepLR':
        scheduler_ = lr_scheduler.StepLR(
            optimizer=optimizer_, step_size=Scheduler['step_size'], gamma=Scheduler['gamma']
        )

    elif Scheduler['method'] == 'MultiStepLR':
        scheduler_ = lr_scheduler.MultiStepLR(
            optimizer=optimizer_, milestones=Scheduler['milestones'], gamma=Scheduler['gamma']
        )

    elif Scheduler['method'] == 'ExponentialLR':
        scheduler_ = lr_scheduler.ExponentialLR(
            optimizer=optimizer_, gamma=Scheduler['gamma']
        )

    else:
        raise ValueError('scheduler method must be "StepLR" or "MultiStepLR" or "ExponentialLR"')

    # 早停机制配置
    if Early_Stopping['applying']:
        early_stopping_ = early_stopping.EarlyStopping(
            patience=Early_Stopping['patience'],
            min_delta=Early_Stopping['min_delta'],
            verbose=Early_Stopping['verbose']
        )
    else:
        early_stopping_ = None

    # 初始化训练器
    runner_ = runner.RunnerM(
        model=model,
        optimizer=optimizer_,
        metric=metric.accuracy,
        loss_fn=loss_function_,
        batch_size=Runner['batch_size'],
        scheduler=scheduler_
    )
    if Early_Stopping['applying']:
        runner_.early_stopping = early_stopping_

    # 启动训练
    runner_.train(
        train_set=(X_train, y_train),
        dev_set=(X_val, y_val),
        num_epochs=Runner['num_epochs'],
        log_iters=Runner['log_iters'],
        save_dir=save_dir
    )

    # 最终测试
    train_scores = runner_.train_scores
    dev_scores = runner_.dev_scores
    train_loss = runner_.train_loss
    dev_loss = runner_.dev_loss
    test_acc, test_loss = runner_.evaluate((X_test, y_test))
    print(f"\nFinal Test Performance: Loss={test_loss:.4f}, Acc={test_acc:.4f}")
    return test_acc, train_scores, dev_scores, train_loss, dev_loss


def main(directory_, component_, layers_):
    # 文件夹创建
    file_name = os.path.join(directory_['save_dir'], "counter")
    ith = get_ith(file_name)
    directory_['save_dir'] = os.path.join(directory_['save_dir'], f"model_{ith}")
    os.makedirs(directory_['save_dir'], exist_ok=True)

    # 执行训练
    test_acc, _, _, _, _ = cnn_train(layers_, **directory_, **component_)

    # 保存训练数据
    data = {
        'dataset': directory_,
        'component': component_
    }
    js_save("train_set.json", data, directory_['save_dir'])

    # 保存模型配置及准确率
    data = {
        'accuracy': float(test_acc),
        'layers': [l.to_dict() for l in layers_]
    }
    js_save("model.json", data, directory_['save_dir'])


if __name__ == "__main__":
    # 下载配置
    # 目录与数据配置
    directory = {
        # 保存训练结果（如模型权重、日志）的目录路径
        # 类型: str
        # 示例: r".\result\cnn_experiment", "/path/to/output"
        'save_dir': ".\\result\\test_CNN",

        # 数据集所在的根目录路径
        # 类型: str
        # 示例: '.\\dataset\\MNIST', '/data/cifar10'
        'dataset_dir': '.\\dataset\\MNIST',

        # 输入数据的维度信息
        'data_size': {
            # 输入图像的通道数 (例如：1 表示灰度图, 3 表示 RGB 彩色图)
            # 类型: int
            # 典型范围: > 0 (常用 1 或 3)
            'channels': 1,
            # 输入图像的高度（像素）
            # 类型: int
            # 典型范围: > 0
            'height': 28,
            # 输入图像的宽度（像素）
            # 类型: int
            # 典型范围: > 0
            'width': 28
        }
    }

    # 训练组件配置
    component = {
        'Loss_function': {
            # 使用的损失函数名称
            # 类型: str
            # 可选值: 'MultiCrossEntropyLoss'
            'method': 'MultiCrossEntropyLoss'
        },
        'Optimizer': {
            # 使用的优化器算法名称
            # 类型: str
            # 可选值: 'SGD', 'MomentumGD'
            'method': 'MomentumGD',

            # 初始学习率
            # 类型: float
            # 范围: > 0 (例如: 1e-5 到 1.0, 常根据模型和数据集调整)
            'init_lr': 0.1,

            # 优化器特定的参数 (例如 Momentum 的 beta 值)
            # 类型: float
            # 范围: [0, 1) (对于 Momentum 通常 >= 0.8, 如 0.9)
            'init_beta': 0.9,
        },
        'Scheduler': {
            # 学习率调度器名称
            # 类型: str
            # 可选值: 'StepLR', 'MultiStepLR', 'ExponentialLR'
            'method': 'StepLR',

            # StepLR/ExponentialLR: 学习率衰减的周期（以 epoch 为单位）
            # 类型: int
            # 范围: > 0
            'step_size': 9,

            # StepLR/MultiStepLR/ExponentialLR: 学习率衰减的乘法因子
            # 类型: float
            # 范围: (0, 1] (例如: 0.1, 0.3, 0.5, 0.9)
            'gamma': 0.3,

            # MultiStepLR: 在哪些 epoch 进行学习率衰减的列表
            # 类型: list[int] or None
            # 示例: [30, 80, 120]
            'milestones': None
        },
        'Runner': {  # 训练/评估循环配置
            # 每个批次的大小
            # 类型: int
            # 范围: > 0 (通常是 2 的幂，如 16, 32, 64, 128, ..., 取决于 GPU 内存)
            'batch_size': 64,

            # 训练的总轮数 (epochs)
            # 类型: int
            # 范围: > 0 (例如: 10, 50, 100, ...)
            'num_epochs': 1,

            # 每隔多少次迭代 (iterations/batches) 记录一次日志
            # 类型: int
            # 范围: > 0
            'log_iters': 300
        },
        'Early_Stopping': {
            # 是否启用早停法
            # 类型: bool
            # 可选值: True, False
            'applying': False,

            # 在停止前等待多少个没有改善的更新次数，与'log_iters'值相关
            # 类型: int
            # 典型范围: >= 0 (如果 applying=True, 通常 > 5 或 10)
            'patience': 20,

            # 被认为是改善所需的最小变化量 (监控指标：验证集loss)
            # 类型: float
            # 典型范围: >= 0 (例如: 0.0, 1e-4, 0.001)
            'min_delta': 0.0001,

            # 是否在早停时打印消息
            # 类型: bool
            # 可选值: True, False
            'verbose': True
        }
    }

    # 模型配置
    # 按顺序定义网络的各个层。
    # 正则化在每个层中独立实现，可设置参数weight_decay_lambda。 若其值为0，表示不启用正则化；若其值大于0，表示启用正则化
    # 损失函数层可以不在此列表中定义，默认在训练脚本中使用 MultiCrossEntropyLoss，其在cnn_train()函数中配置
    # 可选层级[Conv2D(), MaxPooling2D(), Linear(), Flatten(), MultiCrossEntropyLoss(), ReLU(), Logistic(), Tanh()],
    # 上述各层可选参数具体见"./mynn/op.py"和"./mynn/activation_function.py"中的注释，此处不再赘述
    # 线性层初始化时最好使用'He'或'Xavier'方法，否则模型可能无法正常被优化
    layers = [
        Conv2D(C_in=1, C_out=8, kernel_size=11, stride=1, padding=3, weight_decay_lambda=0),  # 24
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding=0),  # 12

        Conv2D(C_in=8, C_out=64, kernel_size=5, stride=1, padding=1, weight_decay_lambda=0),  # 10
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding=0),  # 5

        Conv2D(C_in=64, C_out=128, kernel_size=3, stride=1, padding=1, weight_decay_lambda=0),  # 5
        ReLU(),

        Flatten(),
        Linear(in_dim=128 * 5 * 5, out_dim=512, initialize_method='HeInit', weight_decay_lambda=0.001),
        ReLU(),
        Linear(in_dim=512, out_dim=64, initialize_method='HeInit', weight_decay_lambda=0.001),
        ReLU(),
        Linear(in_dim=64, out_dim=10, initialize_method='HeInit', weight_decay_lambda=0.001)
    ]

    # 执行主程序
    main(directory, component, layers)

