import pickle
import os
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import seaborn as sns


def visualize_model_params(pkl_path, output_dir=r'.\mlp_weight_plot'):
    with open(pkl_path, 'rb') as f:
        param_list = pickle.load(f)

    layers_params = param_list[2:]

    for layer_idx, layer in enumerate(layers_params, 1):
        W = cp.asnumpy(layer['W'])
        W_normalized = (W - W.min()) / W.ptp()
        W_scaled = (W_normalized * 255).astype(np.uint8)
        W_flat = W.flatten()

        # ===================== 热力图单独保存 =====================
        plt.figure(figsize=(10, 8))
        heatmap = plt.imshow(W_scaled, cmap='viridis',
                             aspect='equal' if W.shape[0] == W.shape[1] else 'auto')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f'Layer {layer_idx} Heatmap\n{W.shape}')
        plt.xlabel('Output Units')
        plt.ylabel('Input Units')
        plt.savefig(os.path.join(output_dir, f'layer{layer_idx}_heatmap.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # ===================== 分布图单独保存 =====================
        plt.figure(figsize=(10, 6))
        sns.histplot(W_flat, kde=True, bins=50, color='royalblue',
                     edgecolor='black', linewidth=0.5)

        mean_val = np.mean(W_flat)
        std_val = np.std(W_flat)
        plt.axvline(mean_val, color='r', linestyle='--', linewidth=1.2)
        plt.axvline(mean_val + std_val, color='g', linestyle=':', linewidth=1)
        plt.axvline(mean_val - std_val, color='g', linestyle=':', linewidth=1)
        plt.annotate(f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                     xy=(0.75, 0.85), xycoords='axes fraction',
                     bbox=dict(boxstyle="round", fc="white", ec="gray"))

        plt.title(f'Layer {layer_idx} Parameter Distribution')
        plt.xlabel('Parameter Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'layer{layer_idx}_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


# 注意需要写入准确的参数文件的地址
visualize_model_params(
    pkl_path=r'D:\python object\neural network\project1\codes_gpu\result\test_MLP\model_11\best_model.pkl',
    output_dir=r'.\mlp_weight_plot'
)
