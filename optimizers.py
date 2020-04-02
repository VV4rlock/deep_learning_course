from base import BaseOptimizer, BaseModel
import numpy as np


class SGD(BaseOptimizer):
    def __init__(self, net: BaseModel, learning_rate, loss_func, label_smoothing=0):
        self.net = net
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.label_smoothing = label_smoothing

    def update_rule(self, dW: np.ndarray):
        return - self.learning_rate * dW

    def update_layers_rule(self, param_list: list) -> list:
        out = []
        for layer_param in param_list:
            temp = []
            for k in layer_param:
                temp.append(self.update_rule(k))
            out.append(temp)
        return out


class Momentum(BaseOptimizer):
    def __init__(self, net: BaseModel, learning_rate, loss_func, label_smoothing=0, gamma=0.9):
        self.net = net
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.vt_1 = None
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def update_rule(self, dW: np.ndarray, vt_1):
        return self.gamma * vt_1 + self.learning_rate * dW

    def update_layers_rule(self, param_list: list) -> list:
        out = []
        vt_vec = []
        for layer_index, layer_param in enumerate(param_list):
            temp = []
            vt_for_layer = []
            for param_index, param in enumerate(layer_param):
                vt = self.update_rule(param, 0 if self.vt_1 is None else self.vt_1[layer_index][param_index])
                vt_for_layer.append(vt)
                temp.append(-vt)
            vt_vec.append(vt_for_layer)
            out.append(temp)
        self.vt_1 = vt_vec
        return out




