from abc import abstractmethod


class Scheduler:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step(self):
        pass


# 阶梯式衰减
class StepLR(Scheduler):
    def __init__(self, optimizer, step_size=10, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            self.optimizer.init_lr *= self.gamma


# 多阶段衰减
class MultiStepLR(Scheduler):
    def __init__(self, optimizer, milestones=None, gamma=0.1):
        super().__init__(optimizer)
        self.milestones = sorted(milestones)
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count in self.milestones:
            self.optimizer.init_lr *= self.gamma


# 指数衰减
class ExponentialLR(Scheduler):
    def __init__(self, optimizer, gamma=0.7):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.optimizer.init_lr *= self.gamma
