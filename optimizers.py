from base import BaseOptimizer, BaseModel
import numpy as np


class SGD(BaseOptimizer):
    def __init__(self, net: BaseModel):
        self.net = net

    def update_rule(self, dW: np.ndarray):
        pass

    def minimise(self, dz_dl):
        pass