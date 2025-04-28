import os
import gzip
import numpy as np
import cupy as cp


def load_mnist_images(filename):
    """加载MNIST图像文件"""
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        # 转换为[0,1]范围的浮点数并展平为784维向量
        return cp.array(data.reshape(-1, 784) / 255.0, dtype=cp.float32)


def load_mnist_labels(filename):
    """加载MNIST标签文件"""
    with gzip.open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return cp.array(labels, dtype=cp.int64)


def load_dataset(data_dir):
    """加载完整MNIST数据集"""
    train_images = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))

    test_images = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_labels = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    # 从训练集划分验证集（10%）
    split_idx = int(0.9 * len(train_images))
    return (
        train_images[:split_idx], train_labels[:split_idx],  # 训练集
        train_images[split_idx:], train_labels[split_idx:],  # 验证集
        test_images, test_labels  # 测试集
    )


def reshape_images(images, channels, height, width):
    return images.reshape(images.shape[0], channels, height, width)

