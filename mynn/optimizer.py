from abc import abstractmethod


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params:
                    param = layer.params[key]
                    grad = layer.grads[key]
                    layer.params[key] = param - self.init_lr * grad


class MomentumGD(Optimizer):
    def __init__(self, model, init_lr, init_beta=0.9):
        super().__init__(init_lr, model)
        self.beta = init_beta
        self.previous_params = {}

        # 初始化历史参数为None（表示初始时无历史值）
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params:
                    param = layer.params[key]
                    self.previous_params[id(param)] = None

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                lr_scale = getattr(layer, 'lr_scale', 1.0)
                current_alpha = self.init_lr * lr_scale  # 动态学习率α_t
                current_beta = self.beta  # 动态动量系数β_t

                for key in layer.params:
                    param = layer.params[key]
                    grad = layer.grads[key]
                    param_id = id(param)
                    prev_param = self.previous_params[param_id]

                    # 计算参数差值：Δw = w^t - w^{t-1}（首次迭代时Δw=0）
                    delta = param - prev_param if prev_param is not None else 0

                    # 根据公式更新参数
                    param_new = param - current_alpha * grad + current_beta * delta

                    # 保存当前参数值作为下一次的w^{t-1}
                    self.previous_params[param_id] = param.copy()  # 深拷贝防止引用问题

                    # 应用更新
                    param[:] = param_new
