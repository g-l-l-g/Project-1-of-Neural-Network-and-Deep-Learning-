import cupy as cp


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        """
        :param patience: 允许连续无改进的epoch数
        :param min_delta: 判定改进的最小阈值
        :param verbose: 是否打印日志
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = cp.Inf
        self.early_stop = False

    def __call__(self, val_loss=cp.Inf):
        # 首次初始化
        if self.best_loss == cp.Inf:
            self._update_state(val_loss)
            return False

        # 判断是否改进
        if val_loss < (self.best_loss - self.min_delta):
            self._update_state(val_loss)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def _update_state(self, val_loss):
        self.best_loss = val_loss

