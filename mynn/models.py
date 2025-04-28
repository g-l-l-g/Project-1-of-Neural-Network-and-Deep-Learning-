import pickle
from op import Layer, Linear, Conv2D, MaxPooling2D, Flatten
from activation_function import ReLU, Logistic, Tanh


class MLP(Layer):
    def __init__(self, size_list=None, act_func=None, lambda_list=None, initialize_method=None):
        super().__init__()
        self.layers = []
        self.size_list = size_list

        self.lambda_list = [lambda_list for _ in range(len(size_list) - 1)] \
            if type(lambda_list) is not list else lambda_list

        self.act_func = [act_func for _ in range(len(size_list) - 2)] if type(act_func) is not list else act_func

        self.initialize_method = [initialize_method for _ in range(len(size_list) - 1)] \
            if type(initialize_method) is not list else initialize_method

        if size_list is not None and act_func is not None:
            for i in range(len(size_list) - 1):

                layer = Linear(
                    in_dim=size_list[i],
                    out_dim=size_list[i+1],
                    initialize_method=self.initialize_method[i],
                    weight_decay_lambda=self.lambda_list[i]
                )

                # 激活函数添加
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(self.activation(i))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, \
            ('Model has not initialized yet. Use model.load_model to load a model or create a new model '
             'with size_list and act_func offered.')
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def activation(self, idx):
        """激活函数识别"""
        if self.act_func[idx] == 'Logistic':
            layer_f = Logistic()
        elif self.act_func[idx] == 'ReLU':
            layer_f = ReLU()
        elif self.act_func[idx] == 'Tanh':
            layer_f = Tanh()
        else:
            raise ValueError('act_func must be either Logistic or ReLU or Tanh')
        return layer_f

    def load_model(self, param_list_path):
        with open(param_list_path, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.layers = []

        num_linear_layers = len(self.size_list) - 1
        for i in range(num_linear_layers):
            layer_params = param_list[i + 2]
            layer = Linear(
                in_dim=self.size_list[i],
                out_dim=self.size_list[i + 1]
            )

            # 加载参数
            layer.params['W'] = layer_params['W'].copy()
            layer.params['b'] = layer_params['b'].copy()
            layer.weight_decay_lambda = layer_params['weight_decay_lambda']
            layer.initialize_method = layer_params['initialize_method']
            self.layers.append(layer)

            # 添加激活函数层（最后一层不添加）
            if i < num_linear_layers - 1:
                layer_f = self.activation(i)
                self.layers.append(layer_f)

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if isinstance(layer, Linear):
                param_list.append({
                    'W': layer.params['W'].copy(),
                    'b': layer.params['b'].copy(),
                    "in_dim": layer.in_dim,
                    "out_dim": layer.out_dim,
                    "initialize_method": layer.initialize_method,
                    "weight_decay_lambda": layer.weight_decay_lambda
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class CNN(Layer):
    def __init__(self, layers=None):
        super().__init__()
        self.layers = layers if layers is not None else []

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, path):
        with open(path, 'rb') as f:
            model_info = pickle.load(f)

        # 重建网络结构（必须与保存顺序严格一致）
        self.layers = []

        for layer_dict in model_info['layers']:
            layer = self.create_layer_from_dict(layer_dict)
            self.layers.append(layer)

        # 加载参数（添加索引验证逻辑）
        saved_params = model_info['params']
        param_idx = 0
        for layer in self.layers:
            if isinstance(layer, (Linear, Conv2D)):
                layer.params['W'] = saved_params[param_idx]['W'].copy()
                layer.params['b'] = saved_params[param_idx]['b'].copy()
                param_idx += 1

    def save_model(self, path):
        """ 保存模型结构与参数 """
        model_info = {
            'layers': [l.to_dict() for l in self.layers],
            'params': []
        }

        # 提取可训练层的参数
        for l in self.layers:
            if isinstance(l, (Linear, Conv2D)):
                model_info['params'].append({
                    'W': l.params['W'].copy(),
                    'b': l.params['b'].copy()
                })

        with open(path, 'wb') as f:
            pickle.dump(model_info, f)

    @ staticmethod
    def create_layer_from_dict(layer_dict):
        """ 根据字典创建层对象 """
        layer_type = layer_dict.pop('type')
        if layer_type == 'Conv2D':
            return Conv2D(**layer_dict)
        elif layer_type == 'MaxPooling2D':
            return MaxPooling2D(**layer_dict)
        elif layer_type == "Linear":
            return Linear(**layer_dict)
        elif layer_type == "Flatten":
            return Flatten()
        elif layer_type == "ReLU":
            return ReLU()
        elif layer_type == "Tanh":
            return Tanh()
        elif layer_type == "Logistic":
            return Logistic()
        else:
            raise ValueError(f"未知的层类型: {layer_type}")
