import os
import json
import sys
import cupy as cp

sys.path.append(r"D:\python object\neural network\project1\codes_gpu\mynn")
from mynn import models, optimizer, metric, lr_scheduler, runner, download_dataset, op, early_stopping


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


def mlp_train(
        size_list=None,
        act_func_list=None,
        lambda_list=None,
        initialize_method=None,
        save_dir=None,
        dataset_dir=None,
        Loss_function=None,
        Optimizer=None,
        Scheduler=None,
        Runner=None,
        Early_Stopping=None,
) -> cp.ndarray:

    # 数据加载与预处理
    X_train, y_train, X_val, y_val, X_test, y_test = download_dataset.load_dataset(dataset_dir)

    # 模型配置
    model = models.MLP(
        size_list=size_list,
        act_func=act_func_list,
        lambda_list=lambda_list,
        initialize_method=initialize_method,
    )

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
    return model, test_acc, train_scores, dev_scores, train_loss, dev_loss


def main(dir_, component_, layers_):
    # 文件夹创建
    file_name = os.path.join(dir_['save_dir'], "counter")
    ith = get_ith(file_name)
    dir_['save_dir'] = os.path.join(dir_['save_dir'], f"model_{ith}")
    os.makedirs(dir_['save_dir'], exist_ok=True)

    # 执行训练
    model, test_acc, _, _, _, _ = mlp_train(**layers_, **dir_, **component_)

    # 保存训练数据
    data = {
        'dataset': dir_,
        'component': component_
    }
    js_save("train_set.json", data, dir_['save_dir'])

    # 保存模型配置及准确率
    data = {
        'accuracy': float(test_acc),
        'layers': [l.to_dict() for l in model.layers]
    }
    js_save("model.json", data, dir_['save_dir'])


if __name__ == "__main__":
    # 目录配置
    directory = {
        # 保存训练结果（如模型权重、日志）的目录路径
        # 类型: str
        # 示例: r"./result/my_experiment", "/path/to/output"
        'save_dir': r"./result/test_MLP",

        # 数据集所在的根目录路径
        # 类型: str
        # 示例: './dataset/MNIST', '/data/imagenet'
        'dataset_dir': './dataset/MNIST'
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
            # 范围: > 0 (例如: 1e-5 到 1.0)
            'init_lr': 0.3,

            # 优化器特定的参数 (例如 Momentum 的 beta 值)
            # 类型: float
            # 范围: [0, 1) (对于 Momentum 通常 >= 0.8)
            'init_beta': 0.9
        },
        'Scheduler': {
            # 学习率调度器名称
            # 类型: str
            # 可选值: 'StepLR', 'MultiStepLR', 'ExponentialLR'
            'method': 'StepLR',

            # StepLR/ExponentialLR: 学习率衰减的周期（以 epoch 为单位）
            # 类型: int
            # 范围: > 0
            'step_size': 4,

            # StepLR/MultiStepLR/ExponentialLR: 学习率衰减的乘法因子
            # 类型: float
            # 范围: (0, 1]
            'gamma': 0.5,

            # MultiStepLR: 在哪些 epoch 进行学习率衰减的列表
            # 类型: list[int] or None
            # 示例: [30, 80, 120]
            'milestones': None
        },
        'Runner': {  # 训练/评估循环配置
            # 每个批次的大小
            # 类型: int
            # 范围: > 0 (通常是 2 的幂，如 16, 32, 64, 128, ...)
            'batch_size': 16,

            # 训练的总轮数 (epochs)
            # 类型: int
            # 范围: > 0
            'num_epochs': 10,

            # 每隔多少次迭代 (iterations/batches) 更新模型并记录一次日志
            # 类型: int
            # 范围: > 0
            'log_iters': 200
        },
        'Early_Stopping': {
            # 是否启用早停法
            # 类型: bool
            # 可选值: True, False
            'applying': False,

            # 在停止前等待多少个没有改善的 epoch
            # 类型: int
            # 范围: >= 0 (如果 applying=True, 通常 > 0)
            'patience': 20,

            # 被认为是改善所需的最小变化量 (监控指标，如 loss 或 accuracy)
            # 类型: float
            # 范围: >= 0 (例如: 0.0, 1e-4, 0.001)
            'min_delta': 0.0005,

            # 是否在早停时打印消息
            # 类型: bool
            # 可选值: True, False
            'verbose': True
        }
    }

    # 模型层参数配置 (示例针对 MLP)
    layers = {
        # 定义网络各层的大小（神经元数量），包括输入层和输出层
        # 类型: list[int]
        # 规则: 列表长度 >= 2，所有元素 > 0。第一个是输入维度，最后一个是输出维度。
        # 示例: [784, 512, 10] (输入784, 隐藏层512, 输出10)
        'size_list': [784, 1024, 256, 64, 10],

        # 用于隐藏层的激活函数名称。可以是单个字符串（应用于所有隐藏层）或列表（逐层指定）
        # 类型: str or list[str]
        # 可选值: 'ReLU', 'Sigmoid', 'Tanh'
        'act_func_list': 'Tanh',

        # 正则化强度 (例如 L2 权重衰减的 lambda)。可以是单个值（全局）或列表（逐层）
        # 类型: float or list[float]
        # 典型范围: >= 0 (0 表示无正则化)
        'lambda_list': 0,

        # 权重初始化方法的名称。可以是单个字符串（全局）或列表（逐层）
        # 类型: str or list[str]
        # 可选值: 'HeInit', 'XavierInit', 'GaussianRandomInit', 'UniformRandomInit' 
        'initialize_method': 'HeInit',
    }

    # 执行主程序
    main(directory, component, layers)
