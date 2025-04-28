import cupy as cp


class Initializer:
    """基类，定义初始化接口"""
    @staticmethod
    def initialize(shape: tuple) -> cp.ndarray:
        raise NotImplementedError


class XavierInit(Initializer):
    """
    Xavier 初始化（适用于 Tanh 激活）
    数学原理：权重方差应满足 Var(W) = 2/(fan_in + fan_out)[1,3](@ref)
    """
    @staticmethod
    def initialize(shape):
        fan_in, fan_out = shape[0], shape[1]
        scale = cp.sqrt(2.0 / (fan_in + fan_out))
        return cp.random.normal(0, scale, size=shape)


class HeInit(Initializer):
    """
    He 初始化（适用于 ReLU 激活）
    数学原理：权重方差应满足 Var(W) = 2/fan_in[3,7](@ref)
    """
    @staticmethod
    def initialize(shape):
        fan_in = shape[0]
        scale = cp.sqrt(2.0 / fan_in)
        return cp.random.normal(0, scale, size=shape)


class GaussianRandomInit(Initializer):
    """
    高斯分布随机初始化（默认标准差0.01）
    适用场景：简单初始化或配合后续标准化层[8,10](@ref)
    """
    @staticmethod
    def initialize(shape, std=0.01):
        return cp.random.normal(loc=0.0, scale=std, size=shape)


class UniformRandomInit(Initializer):
    """
    均匀分布随机初始化（默认范围[-0.1, 0.1]）
    适用场景：替代简单零初始化[8,10](@ref)
    """
    @staticmethod
    def initialize(shape, limit=0.1):
        return cp.random.uniform(low=-limit, high=limit, size=shape)
