import cupy as cp
import sys
import numpy as np
import torch
sys.path.append(r"/project1/codes_gpu/mynn")
from mynn.op import im2col


def test_performance():
    # 创建512x512的随机张量
    large_tensor = cp.random.randn(16, 1, 28, 28)

    # 使用CuPy事件计时
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()

    result = im2col(large_tensor, 3, 1, 1, 28, 28)

    end.record()
    end.synchronize()
    print(f"执行时间：{cp.cuda.get_elapsed_time(start, end)}ms")

    # 验证输出尺寸
    assert result.shape == (16 * 28 ** 2, 1 * 9), "大尺寸输出形状错误"


def compare_with_pytorch():
    # 创建相同输入数据
    np_input = np.random.randn(2, 3, 32, 32)
    cp_tensor = cp.array(np_input)
    th_tensor = torch.from_numpy(np_input)

    # PyTorch展开操作
    unfolded = th_tensor.unfold(2, 3, 1).unfold(3, 3, 1)
    unfolded = unfolded.permute(0, 2, 3, 1, 4, 5).contiguous()
    pytorch_result = unfolded.view(2 * 30 * 30, 3 * 3 * 3)

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    # 转换函数结果
    custom_result = im2col(cp_tensor, 3, 1, 0, 30, 30).get()

    end.record()
    end.synchronize()
    print(f"执行时间：{cp.cuda.get_elapsed_time(start, end)}ms")

    # 允许1e-5的误差
    np.testing.assert_allclose(custom_result, pytorch_result.numpy(),
                               rtol=1e-5, err_msg="与PyTorch结果不一致")


if __name__ == "__main__":
    print("规格检验")
    test_performance()
    print("pytorch比较")
    compare_with_pytorch()
