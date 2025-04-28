# 仅可视化卷积核图像
import os
import pickle
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np


def visualize_cnn_kernels(pkl_path, output_dir='./cnn_kernels'):
    os.makedirs(output_dir, exist_ok=True)

    with open(pkl_path, 'rb') as f:
        model_info = pickle.load(f)

    params = model_info['params']
    layers = model_info['layers']

    conv_layer_idx = 0
    param_idx = 0

    for layer_dict in layers:
        layer_type = layer_dict.get('type')

        if layer_type == 'Conv2D':
            # 提取卷积核参数
            W = cp.asnumpy(params[param_idx]['W'])
            param_idx += 1

            # 获取维度信息 (输出通道, 输入通道, 高, 宽)
            out_channels, in_channels, kH, kW = W.shape

            # 为每个输出通道创建可视化
            for oc in range(out_channels):
                # 动态计算子图布局
                n_cols = min(in_channels, 8)
                n_rows = int(np.ceil(in_channels / n_cols))

                # 创建子图并确保axes总是一维数组
                fig, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(n_cols * 1.5, n_rows * 1.5))
                axes = np.array(axes).ravel()  # 关键修复：强制转换为一维数组

                # 绘制每个输入通道的卷积核
                for ic in range(in_channels):
                    kernel = W[oc, ic]
                    # 独立归一化
                    vmin, vmax = kernel.min(), kernel.max()
                    norm_kernel = (kernel - vmin) / (vmax - vmin + 1e-8)

                    # 绘制并设置格式
                    axes[ic].imshow(norm_kernel,
                                    cmap='viridis',
                                    vmin=0, vmax=1)
                    axes[ic].set_title(f'In{ic + 1}', fontsize=6)
                    axes[ic].axis('off')

                # 隐藏多余子图
                for ax in axes[in_channels:]:
                    ax.axis('off')

                # 保存图片
                plt.savefig(os.path.join(output_dir,
                                         f'conv{conv_layer_idx + 1}_out{oc + 1}.png'),
                            bbox_inches='tight', dpi=150)
                plt.close()

            conv_layer_idx += 1

        elif layer_type == 'Linear':
            # 跳过全连接层参数
            param_idx += 1


# 使用示例（替换为实际模型路径）
visualize_cnn_kernels(
    pkl_path=r'D:\python object\neural network\project1\codes_gpu\result\test_CNN\model_29\best_model.pkl',
    output_dir=r'.\cnn_weight_plot'
)
