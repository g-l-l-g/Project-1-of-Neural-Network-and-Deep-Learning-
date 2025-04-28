import torch
import cupy as cp
import numpy as np
import sys
sys.path.append(r"D:\python object\neural network\project1\codes_gpu\mynn")
import op

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)
cp.random.seed(42)


def test_conv2d_backward():
    # 参数设置
    batch_size, C_in, C_out = 16, 1, 5
    H_in, W_in = 28, 28
    kernel_size, stride, padding = 3, 1, 1

    # 自定义 Conv2D
    conv_custom = op.Conv2D(C_in, C_out, kernel_size, stride, padding)
    # PyTorch Conv2d
    conv_torch = torch.nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True)

    # 同步权重
    W_cp = conv_custom.params['W']   # cupy
    b_cp = conv_custom.params['b']   # cupy
    conv_torch.weight.data = torch.from_numpy(cp.asnumpy(W_cp)).float()
    conv_torch.bias.data = torch.from_numpy(cp.asnumpy(b_cp)).float()

    # 随机输入
    X_np = np.random.randn(batch_size, C_in, H_in, W_in).astype(np.float32)
    X_torch = torch.tensor(X_np, requires_grad=True)
    X_cp = cp.array(X_np)

    # 前向
    out_custom = conv_custom.forward(X_cp)
    out_torch = conv_torch(X_torch)

    # 随机上游梯度
    grad_out_np = np.random.randn(*out_custom.shape).astype(np.float32)
    grad_out_torch = torch.tensor(grad_out_np)
    grad_out_cp = cp.array(grad_out_np)

    # 自定义反向
    grad_input_custom = conv_custom.backward(grad_out_cp)
    dW_custom = conv_custom.grads['W']
    db_custom = conv_custom.grads['b']

    # PyTorch 反向
    out_torch.backward(grad_out_torch)
    grad_input_torch = X_torch.grad.detach().numpy()
    dW_torch = conv_torch.weight.grad.detach().numpy()
    db_torch = conv_torch.bias.grad.detach().numpy()

    # 比较
    assert np.allclose(cp.asnumpy(grad_input_custom), grad_input_torch, atol=1e-5), \
        f"Input grad mismatch: max diff {np.max(np.abs(cp.asnumpy(grad_input_custom)-grad_input_torch))}"
    assert np.allclose(cp.asnumpy(dW_custom), dW_torch, atol=1e-5), \
        f"Weight grad mismatch: max diff {np.max(np.abs(cp.asnumpy(dW_custom)-dW_torch))}"
    assert np.allclose(cp.asnumpy(db_custom), db_torch.flatten(), atol=1e-5), \
        f"Bias grad mismatch: max diff {np.max(np.abs(cp.asnumpy(db_custom)-db_torch.flatten()))}"

    print("Backward pass gradients match PyTorch!")
    print(f"Input grad error: max diff {np.max(np.abs(cp.asnumpy(grad_input_custom)-grad_input_torch))}")
    print(f"Weight grad error: max diff {np.max(np.abs(cp.asnumpy(dW_custom)-dW_torch))}")
    print(f"Bias grad error: max diff {np.max(np.abs(cp.asnumpy(db_custom)-db_torch.flatten()))}")


if __name__ == '__main__':
    test_conv2d_backward()
