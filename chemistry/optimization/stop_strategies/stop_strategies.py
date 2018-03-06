import numpy as np
from chemistry.utils import linalg


class BaseStopStrategy:
    def __and__(self, other):
        return StopStrategyAnd(self, other)

    def __or__(self, other):
        return StopStrategyOr(self, other)


class StopStrategyAnd(BaseStopStrategy):
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __call__(self, **kwargs):
        return self.first(**kwargs) and self.second(**kwargs)


class StopStrategyOr(BaseStopStrategy):
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __call__(self, **kwargs):
        return self.first(**kwargs) or self.second(**kwargs)


class GradNorm(BaseStopStrategy):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, grad, **kwargs):
        return np.linalg.norm(grad) < self.threshold


class SaddlePointType(BaseStopStrategy):
    def __init__(self, n_negatives):
        self.n_negatives = n_negatives

    def __call__(self, hess, **kwargs):
        negatives = 0
        for value in linalg.calc_singular_values(hess):
            if value < 0:
                negatives += 1
        return negatives == self.n_negatives
