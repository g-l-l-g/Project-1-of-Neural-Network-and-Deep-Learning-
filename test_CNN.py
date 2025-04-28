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
    directory = {
        'save_dir': ".\\result\\test_CNN",
        'dataset_dir': '.\\dataset\\MNIST',
        'data_size': {
            'channels': 1,
            'height': 28,
            'width': 28
        }
    }

    # 训练组件配置
    component = {
        'Loss_function': {
            'method': 'MultiCrossEntropyLoss'
        },
        'Optimizer': {
            'method': 'MomentumGD',
            'init_lr': 0.1,
            'init_beta': 0.9,
        },
        'Scheduler': {
            'method': 'StepLR',
            'step_size': 9,
            'gamma': 0.3,
            'milestones': None
        },
        'Runner': {
            'batch_size': 16,
            'num_epochs': 20,
            'log_iters': 100
        },
        'Early_Stopping': {
            'applying': False,
            'patience': 20,
            'min_delta': 0.0001,
            'verbose': True
        }
    }

    # 模型配置
    # 正则化在每个层中独立实现，可设置参数weight_decay_lambda。 若其值为0，表示不启用正则化；若其值大于0，表示启用正则化
    # 不在layers中显式配置损失函数层，默认使用交叉熵损失函数，配置在cnn_train()函数中
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

