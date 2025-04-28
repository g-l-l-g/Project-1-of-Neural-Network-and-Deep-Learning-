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
    # 下载配置
    directory = {
        'save_dir': r"./result/test_MLP",
        'dataset_dir': './dataset/MNIST'
    }

    # 训练组件配置
    component = {
        'Loss_function': {
            'method': 'MultiCrossEntropyLoss'
        },
        'Optimizer': {
            'method': 'MomentumGD',
            'init_lr': 0.3,
            'init_beta': 0.9
        },
        'Scheduler': {
            'method': 'StepLR',
            'step_size': 4,
            'gamma': 0.5,
            'milestones': None
        },
        'Runner': {
            'batch_size': 16,
            'num_epochs': 10,
            'log_iters': 200
        },
        'Early_Stopping': {
            'applying': False,
            'patience': 20,
            'min_delta': 0.0005,
            'verbose': True
        }
    }

    # 参数配置
    layers = {
        'size_list': [784, 1024, 256, 64, 10],
        'act_func_list': 'Tanh',
        'lambda_list': 0,
        'initialize_method': 'HeInit',
    }

    # 执行主程序
    main(directory, component, layers)
