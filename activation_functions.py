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
        t = self(input)
        return t * (1-t)

class Hyperbolic(BaseActivationFunction):
    def __call__(self, input: np.ndarray) -> np.ndarray:
        return np.sinh(input)/np.cosh(input)

    def derivate(self, input: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(input) ** 2


if __name__ == "__main__":
    assert isinstance(Linear(), BaseActivationFunction)
    print(isinstance(Linear(), BaseActivationFunction))