import os
import matplotlib.pyplot as plt
import cupy as cp
import sys
sys.path.append(r"D:\python object\neural network\project1\codes_gpu")
from test_CNN import cnn_train
from mynn.op import Linear, Conv2D, MaxPooling2D, Flatten
from mynn.activation_function import ReLU, Tanh, Logistic


def plot_metrics(train_loss, dev_loss, dev_scores, train_set_size, batch_size, log_iters, num_epochs, save_dir):
    """分别绘制三个独立图像"""
    # 转换 CuPy 数组为 NumPy
    train_loss = [cp.asnumpy(x) for x in train_loss]
    dev_loss = [cp.asnumpy(x) for x in dev_loss]
    dev_scores = [cp.asnumpy(x) for x in dev_scores]

    num_iter_per_epoch = (train_set_size + batch_size - 1) // batch_size
    records_per_epoch = (num_iter_per_epoch - 1) // log_iters + 1

    # 计算每个epoch结束对应的记录点索引
    epoch_markers = [(epoch + 1) * records_per_epoch - 1 for epoch in range(num_epochs)]
    label_positions = [epoch * records_per_epoch + records_per_epoch // 2 for epoch in range(num_epochs)]

    # 绘制训练集损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss', color='royalblue')

    for x in epoch_markers:
        if x < len(train_loss):
            plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.xticks(label_positions, [f'Epoch {e + 1}' for e in range(num_epochs)],
               rotation=45, ha='right', fontsize=8)
    plt.xlabel('Training Progress')
    plt.ylabel('Loss')
    plt.title('Train Loss with Epoch Markers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_loss.png'), dpi=300)
    plt.close()

    # 绘制验证集损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(dev_loss, label='Dev Loss', color='royalblue')

    for x in epoch_markers:
        if x < len(dev_loss):
            plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.xticks(label_positions, [f'Epoch {e + 1}' for e in range(num_epochs)],
               rotation=45, ha='right', fontsize=8)
    plt.xlabel('Training Progress')
    plt.ylabel('Loss')
    plt.title('Dev Loss with Epoch Markers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'val_loss.png'), dpi=300)
    plt.close()

    # 绘制验证集准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(dev_scores, label='Dev accuracy', color='royalblue')

    for x in epoch_markers:
        if x < len(dev_scores):
            plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.xticks(label_positions, [f'Epoch {e + 1}' for e in range(num_epochs)],
               rotation=45, ha='right', fontsize=8)
    plt.xlabel('Training Progress')
    plt.ylabel('accuracy')
    plt.title('Dev accuracy with Epoch Markers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'val_accuracy.png'), dpi=300)
    plt.close()


def main(directory_, component_, layers_):
    # 执行训练，保存训练结果
    _, train_scores, dev_scores, train_loss, dev_loss = cnn_train(layers_, **directory_, **component_)

    plot_metrics(train_loss, dev_loss, dev_scores,
                 54000,
                 component_['Runner']["batch_size"],
                 component_['Runner']["log_iters"],
                 component_['Runner']["num_epochs"],
                 directory_["save_dir"])


if __name__ == "__main__":
    # 下载配置
    directory = {
        'save_dir': r".\cnn_train_plot",
        'dataset_dir': r'D:\python object\neural network\project1\codes_gpu\dataset\MNIST',
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
            'step_size': 4,
            'gamma': 0.3,
            'milestones': None
        },
        'Runner': {
            'batch_size': 16,
            'num_epochs': 10,
            'log_iters': 100
        },
        'Early_Stopping': {
            'applying': False,
            'patience': 20,
            'min_delta': 0.0001,
            'verbose': True
        }
    }

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
