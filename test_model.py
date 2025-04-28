import sys
sys.path.append(r"D:\python object\neural network\project1\codes_gpu\mynn")
from mynn import models, download_dataset, metric


def model_test(dataset_dir=None, model_dir=None, model_type='CNN', channels=1, height=28, width=28):
    # 加载已训练好的模型
    if model_type == 'CNN':
        model = models.CNN()
    elif model_type == 'MLP':
        model = models.MLP()
    else:
        raise ValueError('model_type must be either CNN or MLP')
    model.load_model(model_dir)

    # 下载测试集
    _, _, _, _, test_images, test_labels = download_dataset.load_dataset(dataset_dir)

    # 模型推理
    if model_type == 'CNN':
        test_images = download_dataset.reshape_images(test_images, channels, height, width)
    elif model_type == 'MLP':
        pass
    else:
        raise ValueError('model_type must be either CNN or MLP')
    logits = model(test_images)

    # 输出测试准确率
    print("Test Accuracy:", metric.accuracy(logits, test_labels))


if __name__ == '__main__':

    # 地址设置
    directory = {
        'dataset_dir': r'.\dataset\MNIST',
        'model_dir': r'D:\python object\neural network\project1\codes_gpu\result\test_CNN\model_18\best_model.pkl'
    }

    # 模型配置
    model_set = {
        'model_type': 'CNN',
        'channels': 1,
        'height': 28,
        'width': 28
    }

    # 执行测试
    model_test(**directory, **model_set)
