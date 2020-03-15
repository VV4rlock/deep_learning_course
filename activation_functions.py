from base import BaseActivationFunction
import numpy as np


class ReLU(BaseActivationFunction):
    def __call__(self, input: np.ndarray) -> np.ndarray:
        return np.where(input < 0, 0, input)

    def derivate(self, input: np.ndarray) -> np.ndarray:
        return np.where(input < 0, 0, 1)


class Linear(BaseActivationFunction):
    def __call__(self, input: np.ndarray) -> np.ndarray:
        return input[:]

    def derivate(self, input: np.ndarray) -> np.ndarray:
        return np.ones(input.shape)


class Sigmoid(BaseActivationFunction):
    def __call__(self, input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))

    def derivate(self, input: np.ndarray) -> np.ndarray:
        pass