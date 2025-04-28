# Project-1-of-Neural-Network-and-Deep-Learning-

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

一个基于 CIFAR-10 数据集的深度学习训练框架，支持模型训练、超参数搜索和可视化分析。

## 🚀 功能特性
- ​**模型训练**：使用`test_CNN.py`进行MLP模型训练，结果保存在目录`/result/test_CNN`下
- ​**模型训练**：使用`test_MLP.py`进行MLP模型训练，结果保存在目录`/result/test_MLP`下
- ​**超参数搜索**：使用 `mlp_hyperparameter_search.py` 搜索MLP的超参数配置，结果保存在目录`/result/test_hyperparameter_search`下
- ​**权重可视化**：在目录`/draw`下,使用 `mlp_weight_plot.py` 和 `cnn_weight_plot.py`生成参数可视化图像，图像分别保存在同名目录下
- **训练过程可视化**：在目录`/draw`下,使用 `mlp_train_plot.py` 和 `cnn_train_plot.py`随训练轮数变化曲线，图像分别保存在同名目录下
- ​**模型测试**： 使用`test_model.py`测试已训练模型在测试集上的准确率

## 📂 项目结构
```
.
├── dataset/                       # 数据集根目录
│
├── draw/                          # 实现权重可视化和训练过程可视化
│
├── mynn/                          # 自定义神经网络模块
│   ├── __init__.py                # 包初始化
│   ├── activation_function.py     # 激活函数实现
│   ├── download_dataset.py        # 数据集下载
│   ├── early_stopping.py          # 早停实现
│   ├── initializer.py             # 参数初始化
│   ├── lr_scheduler.py            # 学习率调度器
│   ├── metric.py                  # 评估指标（准确率）
│   ├── models.py                  # 模型定义
│   ├── op.py                      # 基础算子
│   ├── optimizer.py               # 优化器
│   └── runner.py                  # 训练流程控制
│
├── result/
│   ├── test_CNN                   # 保存CNN训练结果
│   ├── test_MLP                   # 保存MLP训练结果
│   └── test_hyperparameter_search # 保存超参数搜索结果
│
├── mlp_hyperparameter_search.py   # 超参数搜索
├── test_CNN.py                    # CNN训练
├── test_MLP.py                    # MLP训练
├── test_model.py                  # 准确率测试
├── requirements.txt               # 依赖库列表
├── LICENSE                        # MIT许可证
└── README.md                      # 项目文档

```
## 🧠 模型说明
模型定义位于 mynn/ 文件夹中，支持自定义网络结构、激活函数和损失函数等。

## 📦 安装指南
### 环境要求
- Python 3.8+ （本项目使用3.10）
- CUDA 12.5 （项目使用 GPU 加速），下载链接 <https://developer.nvidia.com/cuda-12-5-0-download-archive>, 注意需要下载到C盘中，并添加环境变量（如下图所示），否则可能无法正常运行
-  ![环境变量设置](/img/ev_settings.png)
  
### 快速安装
#### 安装依赖
pip install -r requirements.txt

## 🛠 使用说明
- 超参数配置范围，具体见train.py中model_train()函数定义
- 算子配置见`/mynn/op.py`文件
### 库调用路径修改
- 在`test_MLP.py`，`test_CNN.py`，`test_model.py`，`/draw/mlp_train_plot.py`，`/draw/cnn_train_plot.py`文件的库函数导入模块，需要正确修改库识别路径以能正确导入自定义库mynn
### 训练模型
- MLP：在`test_MLP.py`文件中根据参数可选范围修改字典directory，component，layers中值
- CNN：在`test_CNN.py`文件中根据参数可选范围修改字典directory，component，layers中值
### 超参数搜索
- 修改搜索范围：在`mlp_hyperparameter_search.py`文件中修改字典search_config中各个键的值的范围
### 准确率测试
- 测试不同权重文件：修改`test_model.py`的字典directory的值以设置不同的数据集地址和模型地址，修改字典model_set的值以测试不同类型的模型
### 可视化
- 训练过程可视化：在目录`/draw`下, 运行 `mlp_train_plot.py` 和 `cnn_train_plot.py`，超参数修改类同"训练模型"部分
- 权重可视化：在目录`/draw`下, 运行`mlp_weight_plot.py` 和 `cnn_weight_plot.py`，选定pkl_path值以选取模型，选定output_dir值以确定图像保存地址





