import torch
import cupy as cp
import numpy as np
import time
import sys
sys.path.append(r"D:\python object\neural network\project1\codes_gpu\mynn")
import op

# 固定随机种子
torch.manual_seed(0)
np.random.seed(0)
cp.random.seed(0)

# 参数设置
batch_size, C_in, C_out = 16, 1, 3
H_in, W_in = 28, 28
kernel_size, stride, padding = 3, 1, 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化自定义 Conv2D
conv_custom = op.Conv2D(C_in, C_out, kernel_size, stride, padding)
# 初始化 PyTorch Conv2d
conv_torch = torch.nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding).to(device)

# 将自定义权重复制到 PyTorch 模型
W_cp = conv_custom.params['W']                        # cupy array
b_cp = conv_custom.params['b']
W_cpu = cp.asnumpy(W_cp)
b_cpu = cp.asnumpy(b_cp)
conv_torch.weight.data = torch.from_numpy(W_cpu).float().to(device)
conv_torch.bias.data = torch.from_numpy(b_cpu).float().to(device)

# 随机输入
X_np = np.random.randn(batch_size, C_in, H_in, W_in).astype(np.float32)
X_torch = torch.from_numpy(X_np).to(device)
X_cp = cp.array(X_np)

# 热身
_ = conv_custom(X_cp)
_ = conv_torch(X_torch)


if device == 'cuda':
    start1 = time.time()
    out_torch = conv_torch(X_torch)
    end1 = time.time()
else:
    start1 = time.time()
    out_torch = conv_torch(X_torch)
    end1 = time.time()

start2 = time.time()
out_custom = conv_custom(X_cp)
end2 = time.time()
print(out_custom.shape)

out_custom_np = cp.asnumpy(out_custom)
out_torch_np = out_torch.detach().cpu().numpy()
max_diff = np.max(np.abs(out_custom_np - out_torch_np))

print(f"最大绝对误差: {max_diff:.6e}")
print(f"torch时间：{end1-start1:.6e}")
print(f"custom时间：{end2-start2:.6e}")
