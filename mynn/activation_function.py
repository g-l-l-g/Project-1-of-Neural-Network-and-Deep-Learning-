# 激活函数
from op import Layer
import cupy as cp


class ReLU(Layer):
    """ReLU 激活层"""
    def __init__(self):
        super().__init__()
        self.mask = None  # 保存前向传播中的掩码

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X: cp.ndarray) -> cp.ndarray:
        self.mask = (X > 0)
        return X * self.mask

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        return grad * self.mask


class Logistic(Layer):
    """Logistic 激活层"""
    def __init__(self):
        super().__init__()
        self.output = None

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X: cp.ndarray) -> cp.ndarray:
        self.output = 1 / (1 + cp.exp(-X))
        return self.output

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        return grad * self.output * (1 - self.output)


class Tanh(Layer):
    """Tanh 激活层"""
    def __init__(self):
        super().__init__()
        self.output = None

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X: cp.ndarray) -> cp.ndarray:
        self.output = cp.tanh(X)
        return self.output

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        return grad * (1 - cp.square(self.output))
