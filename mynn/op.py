from abc import ABC, abstractmethod
from initializer import HeInit, GaussianRandomInit, UniformRandomInit, XavierInit
import cupy as cp


class Layer(ABC):
    """
    Abstract base class for neural network layers.

    Subclasses must implement `forward` and `backward` methods.

    Attributes:
        params (dict): Layer parameters (e.g., weights, bias).
        grads (dict): Gradients of parameters computed during backward pass.
    """
    def __init__(self) -> None:
        self.params = {}
        self.grads = {}

    @property
    def optimizable(self) -> bool:
        """Whether the layer has trainable parameters."""
        return len(self.params) > 0

    @abstractmethod
    def forward(self, X: cp.ndarray) -> cp.ndarray:
        """
        Forward pass of the layer.

        Args:
            X (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Output data of shape (batch_size, output_size).
        """
        pass

    @abstractmethod
    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """
        Backward pass of the layer.

        Args:
            grad (np.ndarray): Gradient of the loss w.r.t. the layer output.

        Returns:
            np.ndarray: Gradient of the loss w.r.t. the layer input.
        """
        pass

    def zero_grad(self) -> None:
        """Reset all gradients to zero."""
        for key in self.grads:
            self.grads[key].fill(0)


# 全连接层
class Linear(Layer):
    """
    全连接层（包含权重衰减功能）
    Args:
        in_dim (int): 输入维度
        out_dim (int): 输出维度
        initialize_method (callable): 权重初始化方法（默认np.random.normal）,可选 Xavier 或 He 初始化
        weight_decay_lambda (float): 正则化强度系数（默认1e-8）

    Attributes:
        params (dict): 存储优化参数 {'W': ..., 'b': ...}
            W (np.ndarray): 权重矩阵，形状[in_dim, out_dim]
            b (np.ndarray): 偏置向量，形状[1, out_dim]
        grads (dict): 存储梯度 {'W': ..., 'b': ...}
        input (np.ndarray): 缓存前向传播输入用于反向计算
    """

    def __init__(self, in_dim, out_dim, initialize_method='GaussianRandomInit',
                 weight_decay_lambda=0, lr_scale=1.0) -> None:

        super().__init__()

        # 参数初始化
        size = (in_dim, out_dim)
        # supported_methods = ['HeInit', 'GaussianRandomInit', 'UniformRandomInit', 'XavierInit']
        if initialize_method == 'HeInit':
            self.params['W'] = HeInit.initialize(size)
        elif initialize_method == 'GaussianRandomInit':
            self.params['W'] = GaussianRandomInit.initialize(size)
        elif initialize_method == 'UniformRandomInit':
            self.params['W'] = UniformRandomInit.initialize(size)
        elif initialize_method == 'XavierInit':
            self.params['W'] = XavierInit.initialize(size)

        self.params['b'] = cp.random.normal(size=(1, out_dim))

        # 将参数和梯度存入父类字典
        self.grads['W'] = cp.zeros_like(self.params['W'])
        self.grads['b'] = cp.zeros_like(self.params['b'])

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.initialize_method = initialize_method
        self.weight_decay_lambda = weight_decay_lambda
        self.lr_scale = lr_scale
        self.input = None

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "initialize_method": self.initialize_method,
            "weight_decay_lambda": self.weight_decay_lambda
        }

    def __call__(self, X) -> cp.ndarray:
        return self.forward(X)

    def forward(self, X) -> cp.ndarray:
        """
        执行线性变换: output = X·W + b
            Args:
                X (np.ndarray): 输入数据，形状[batch_size, in_dim]
            Returns:
                np.ndarray: 输出数据，形状[batch_size, out_dim]
        """
        self.input = X
        return cp.dot(X, self.params['W']) + self.params['b']

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """
        计算参数梯度并返回输入梯度
        Args:
            grad (np.ndarray): 来自下一层的梯度，形状[batch_size, out_dim]
        Returns:
            np.ndarray: 传递给上一层的梯度，形状[batch_size, in_dim]
        """
        # 批的数量
        batch_size = grad.shape[0]

        # 计算梯度
        self.grads['W'] = cp.dot(self.input.T, grad) / batch_size  # 平均梯度
        self.grads['b'] = cp.sum(grad, axis=0, keepdims=True) / batch_size

        if self.weight_decay_lambda:
            self.grads['W'] += self.weight_decay_lambda * self.params['W']

        return cp.dot(grad, self.params['W'].T)


# 交叉熵损失
class MultiCrossEntropyLoss(Layer):
    """
    Cross-entropy loss with optional built-in Softmax layer (for numerical stability).
    Use `cancel_softmax()` to disable the built-in Softmax when your model already includes one.

    Args:
        model (Layer): Reference to the neural network model (for backpropagation).
    """

    def __init__(self, model=None) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = True
        self.probabilities = None
        self.labels_one_hot = None
        self.batch_size = None

    def __call__(self, predicts, labels):
        return self.forward((predicts, labels))

    def forward(self, predicts_and_labels):
        """
        Args:
            predicts_and_labels: 元组包含:
                predicts (np.ndarray): 模型输出 [batch_size, num_classes]
                labels (np.ndarray): 真实标签 [batch_size, ] 或 [batch_size, num_classes]
        Returns:
            loss (float): 平均交叉熵损失
        """
        predicts, labels = predicts_and_labels
        self.batch_size = predicts.shape[0]

        # 将标签转换为one-hot编码（如果是类别索引）
        if labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1):
            num_classes = predicts.shape[1]
            self.labels_one_hot = cp.eye(num_classes)[labels.reshape(-1)]
        else:
            self.labels_one_hot = labels.copy()

        # 内置Softmax（若未取消）
        if self.has_softmax:
            shifted_logits = predicts - cp.max(predicts, axis=1, keepdims=True)
            exp = cp.exp(shifted_logits)
            self.probabilities = exp / cp.sum(exp, axis=1, keepdims=True)
        else:
            self.probabilities = predicts.copy()

        # 计算交叉熵损失
        eps = 1e-8  # 防止log(0)
        loss = -cp.sum(self.labels_one_hot * cp.log(self.probabilities + eps)) / self.batch_size
        return loss

    def backward(self, grad_output=1.0):
        """
        计算梯度并传递给模型
        """
        if self.grads is None:
            self.grads = {}

        # 梯度计算
        if self.has_softmax:
            # 当包含Softmax时，梯度为 (prob - y_true)
            grad = (self.probabilities - self.labels_one_hot) / self.batch_size
        else:
            # 当无Softmax时，直接计算交叉熵梯度 (predictions - y_true)/predictions
            # 注意：此处假设输入已经是概率分布，实际情况可能需要根据具体实现调整
            grad = (self.probabilities - self.labels_one_hot) / self.batch_size

        # 将梯度传递给模型进行反向传播
        if self.model is not None:
            self.model.backward(grad)

    def cancel_softmax(self):
        """禁用内置的Softmax层（当模型已包含Softmax时使用）"""
        self.has_softmax = False
        return self


def softmax(X):
    x_max = cp.max(X, axis=1, keepdims=True)
    x_exp = cp.exp(X - x_max)
    partition = cp.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition


class Conv2D(Layer):
    def __init__(self, C_in=None, C_out=None, kernel_size=None,
                 stride=1, padding=0,  weight_decay_lambda=0, lr_scale=1.0):
        super().__init__()  # 启用父类初始化
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_decay_lambda = weight_decay_lambda
        self.lr_scale = lr_scale
        self.input_shape = None
        self.cache = None
        self.H_out = None
        self.W_out = None

        self.params['W'] = cp.random.normal(
            loc=0,
            scale=cp.sqrt(2.0 / (C_in * kernel_size * kernel_size)),
            size=(C_out, C_in, kernel_size, kernel_size)
        )
        self.params['b'] = cp.zeros(C_out)
        self.grads['W'] = cp.zeros_like(self.params['W'])
        self.grads['b'] = cp.zeros_like(self.params['b'])

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "C_in": self.C_in,
            "C_out": self.C_out,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "weight_decay_lambda": self.weight_decay_lambda
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        :param X: cp.ndarray(batch_size, C_in, H_in, W_in)
        :return: cp.ndarray(batch_size, C_out, H_in, W_in)
        """
        # X: cp.ndarray(batch_size, C_in, H_in, W_in)
        # 需要相乘的矩阵为W:(C_in, C_out, kernel_size, kernel_size)
        # 目标为 out:(batch_size, C_out, H_out, W_out)
        # 将X,W展开为所需的二维矩阵形状再相乘，得到out，再将out转为对应的四维张量
        # W_col:(C_out, C_in * kernel_size * kernel_size)
        # X_col:(batch_size * H_out * W_out, C_in * kernel_size * kernel_size)

        batch_size, C_in, H_in, W_in = X.shape
        self.input_shape = X.shape
        self.H_out = (H_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.W_out = (W_in + 2 * self.padding - self.kernel_size) // self.stride + 1

        # reshape
        X_col = im2col(X, self.kernel_size, self.stride, self.padding, self.H_out, self.W_out)
        W_col = self.params['W'].reshape(self.C_out, -1)

        # 前向计算
        out_col = cp.dot(X_col, W_col.T) + self.params['b'][None, :]

        # reshape 回输出张量
        out = out_col.reshape(batch_size, self.H_out, self.W_out, self.C_out)
        out = out.transpose(0, 3, 1, 2)

        # 缓存用于 backward
        self.cache = (X, X_col)
        return out

    def backward(self, grad):
        # grad: (N, C_out, H_out, W_out)
        X, X_col = self.cache
        ks, s, p = self.kernel_size, self.stride, self.padding

        # 将 grad 展平与 im2col 对齐
        grad_col = grad.transpose(0, 2, 3, 1).reshape(-1, self.C_out)  # (N*H_out*W_out, C_out)

        # 梯度 w.r.t. 权重和偏置
        dW_col = grad_col.T.dot(X_col)                                # (C_out, C_in*ks*ks)
        dW = dW_col.reshape(self.params['W'].shape)
        db = grad_col.sum(axis=0)                                      # (C_out,)

        # 应用权重衰减
        if self.weight_decay_lambda:
            dW += self.weight_decay_lambda * self.params['W']

        self.grads['W'] = dW
        self.grads['b'] = db

        # 梯度 w.r.t. 输入
        W_col = self.params['W'].reshape(self.C_out, -1)
        dX_col = grad_col.dot(W_col)                                   # (N*H_out*W_out, C_in*ks*ks)
        dX = col2im(dX_col, self.input_shape, ks, s, p, self.H_out, self.W_out)
        return dX


class MaxPooling2D(Layer):
    def __init__(self, pool_size=2, stride=1, padding=0):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride or self.pool_size  # 默认步长等于池化尺寸
        self.padding = padding
        self.max_mask = None  # 记录最大值位置
        self.input_shape = None
        self.H_out = None
        self.W_out = None

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "pool_size": self.pool_size,
            "stride": self.stride,
            "padding": self.padding
        }

    def __call__(self, X) -> cp.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        前向传播（独立处理每个通道）
        Args:
            X (cp.ndarray): 输入数据，(N, C, H, W)
        Returns:
            cp.ndarray: 池化后数据，(N, C, H_out, W_out)
        """
        self.input_shape = X.shape
        N, C, H, W = X.shape

        # 计算输出尺寸
        self.H_out = (H + 2 * self.padding - self.pool_size) // self.stride + 1
        self.W_out = (W + 2 * self.padding - self.pool_size) // self.stride + 1

        # 将通道合并到批次维度 (N*C, 1, H, W)
        X_reshaped = X.reshape(N * C, 1, H, W)

        # 展开为列矩阵：(N*C*H_out*W_out, pool_size^2)
        cols = im2col(X_reshaped, self.pool_size, self.stride, self.padding, self.H_out, self.W_out)

        # 记录最大值索引并池化
        self.max_mask = cp.argmax(cols, axis=1)
        out = cols[cp.arange(cols.shape[0]), self.max_mask]

        # 恢复形状：(N, C, H_out, W_out)
        out = out.reshape(N, C, self.H_out, self.W_out)
        return out

    def backward(self, d_out):
        """
        反向传播（梯度仅传递到最大值位置）
        Args:
            d_out (cp.ndarray): 上游梯度，(N, C, H_out, W_out)
        Returns:
            cp.ndarray: 输入梯度，(N, C, H, W)
        """
        N, C, H_out, W_out = d_out.shape

        # 梯度展平
        d_out_flat = d_out.reshape(N * C * H_out * W_out)

        # 创建空梯度矩阵
        d_cols = cp.zeros((d_out_flat.size, self.pool_size * self.pool_size))
        d_cols[cp.arange(d_out_flat.size), self.max_mask] = d_out_flat

        # 合并通道后的输入形状
        merged_shape = (N * C, 1) + self.input_shape[2:]

        # 转换回输入梯度
        dX_merged = col2im(d_cols, merged_shape, self.pool_size, self.stride, self.padding, self.H_out, self.W_out)
        return dX_merged.reshape(self.input_shape)


# 此处tensor仅用于表示张量名，没有使用TensorFlow库
def im2col(tensor, kernel_size, stride, padding, H_out, W_out):
    """
    :param tensor: (N, C, H_in, W_in)
    :param kernel_size: (int)
    :param stride: (int)
    :param padding: (int)
    :param H_out: (int)
    :param W_out: (int)
    :return: output_matrix: (N * H_out * W_out, C_in * kernel_size * kernel_size)
    """
    N, C, H_in, W_in = tensor.shape
    if padding > 0:
        tensor = cp.pad(
            array=tensor,
            pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )

    # 提高性能，舍弃for循环方法，使用滑动窗口
    '''out = cp.zeros(shape=(N, H_out*W_out, C*kernel_size*kernel_size))
    for i in range(N):
        for j in range(H_out*W_out):
            h_pos = (j // H_out) * stride
            w_pos = (j % W_out) * stride
            window = tensor[i, :, h_pos:h_pos + kernel_size, w_pos:w_pos + kernel_size]
            out[i, j] = window.ravel()
    return out.reshape(N * H_out * W_out, C*kernel_size*kernel_size)'''

    h_base = cp.arange(H_out) * stride
    w_base = cp.arange(W_out) * stride

    h_grid = h_base[:, None, None, None] + cp.arange(kernel_size)[None, None, :, None]
    w_grid = w_base[None, :, None, None] + cp.arange(kernel_size)[None, None, None, :]

    windows = tensor[:, :, h_grid, w_grid]  # (N, C, H_out, W_out, ks, ks)

    return windows.transpose(0, 2, 3, 1, 4, 5).reshape(-1, C * kernel_size * kernel_size)


def col2im(cols, input_shape, kernel_size, stride, padding, H_out, W_out):
    N, C, H_in, W_in = input_shape

    # 重塑 cols
    cols_reshaped = cols.reshape(N, H_out, W_out, C, kernel_size, kernel_size)
    # 初始化带填充的输入梯度
    H_pad, W_pad = H_in + 2*padding, W_in + 2*padding
    dX_padded = cp.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)

    # 逐位置累加梯度
    for y in range(H_out):
        for x in range(W_out):
            y_start = y * stride
            x_start = x * stride
            dX_padded[:, :, y_start:y_start+kernel_size, x_start:x_start+kernel_size] += cols_reshaped[:, y, x]

    # 去除填充
    if padding > 0:
        return dX_padded[:, :, padding:-padding, padding:-padding]
    return dX_padded


class Flatten(Layer):
    """展平层"""
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
        }

    def __call__(self, X) -> cp.ndarray:
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)
