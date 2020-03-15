from abc import *
import numpy as np


class BaseActivationFunction:
    @abstractmethod
    def __call__(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivate(self, input: np.ndarray) -> np.ndarray:
        pass


class BaseLayerClass:
    @abstractmethod
    def __call__(self, x, phase):
        pass

    def get_previous_layer(self):
        return self._previous_layer

    @abstractmethod
    def get_grad(self):
        pass

    @property
    def is_trainable(self):
        return False

    @abstractmethod
    def backward(self, dy):
        pass

    @abstractmethod
    def update_weights(self, update_func):
        pass

    @abstractmethod
    def get_nrof_trainable_params(self):
        pass

    def get_input_shape(self):
        return self._input_shape

    def get_output_shape(self):
        return self._output_shape


class BaseOptimizer:
    @abstractmethod
    def update_rule(self, dW):
        pass

    @abstractmethod
    def minimise(self, dz_dl):
        pass


class BaseModel:
    @abstractmethod
    def __call__(self, input):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def dump_model(self, filename):
        pass

    @abstractmethod
    def train(self, dataloader, optimiser):
        pass

    @staticmethod
    @abstractmethod
    def load_model(filename):
        pass

