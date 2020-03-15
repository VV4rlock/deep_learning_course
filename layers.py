from base import BaseLayerClass, BaseActivationFunction
from config import *


class InputLayer(BaseLayerClass):
    def __init__(self, shape):
        self._input_shape = shape
        self._output_shape = shape
        self._previous_layer = None

    def __call__(self, x, phase):
        return x

    def is_trainable(self):
        return False

    def backward(self, dy):
        logger.warn("Input layer is not trainable")

    def update_weights(self, update_func):
        logger.warn("Input layer is not trainable")

    def get_grad(self):
        logger.warn("Input layer is not trainable")

    def get_nrof_trainable_params(self):
        return 0


class SoftmaxOutputLayer(BaseLayerClass):
    def __init__(self, previous_layer):
        assert isinstance(previous_layer, BaseLayerClass)
        self._input_shape = previous_layer.get_output_shape()
        self._output_shape = self._input_shape
        self._previous_layer = previous_layer
        self.y = None

    def __call__(self, x, phase):
        e = np.exp(x)
        self.y = e / e.sum()
        return self.y

    def is_trainable(self):
        return False

    def backward(self, dz_dl):
        self._previous_layer.backward(dz_dl)

    def update_weights(self, update_func):
        logger.warn("Softmax layer is not trainable")

    def get_grad(self):
        logger.warn("Softmax layer is not trainable")

    def get_nrof_trainable_params(self):
        return 0


class SimpleOutputLayer(BaseLayerClass):
    def __init__(self, previous_layer):
        assert isinstance(previous_layer, BaseLayerClass)
        self._input_shape = previous_layer.get_output_shape()
        self._output_shape = self._input_shape
        self._previous_layer = previous_layer
        self.y = None

    def __call__(self, x, phase):
        self.y = x[:]
        return x

    def is_trainable(self):
        return False

    def backward(self, dz_dl):
        self._previous_layer.backward(dz_dl)

    def update_weights(self, update_func):
        logger.warn("Output layer is not trainable")

    def get_grad(self):
        logger.warn("Output layer is not trainable")

    def get_nrof_trainable_params(self):
        return 0


class FullyConnectedLayer(BaseLayerClass):
    _distributions = ['uniform', 'normal']
    _sigma_dict = {
        'he': lambda n, m: (2 / n) ** 0.5,
        'xavier': lambda n, m: (1 / n) ** 0.5,
        'glorot': lambda n, m: (1 / (n + m)) ** 0.5
    }

    _l_dict = {
        'he': lambda n, m: (6 / n) ** 0.5,
        'xavier': lambda n, m: (3 / n) ** 0.5,
        'glorot': lambda n, m: (6 / (n + m)) ** 0.5
    }

    def __init__(self, neuron_count, previous_layer, activation_func, init_distribution='normal', initialization='he'):
        assert init_distribution in self._distributions
        assert initialization in self._sigma_dict or initialization in self._sigma_dict
        assert isinstance(previous_layer, BaseLayerClass)
        assert isinstance(activation_func, BaseActivationFunction)
        self._activation_func = activation_func
        self._input_shape = previous_layer.get_output_shape()
        self._output_shape = neuron_count
        self._previous_layer = previous_layer
        self.a = None
        self.dE_da = None
        self.x = None
        if init_distribution == 'normal':
            sigma = self._sigma_dict[initialization](self._input_shape, self._output_shape)
            self.W = np.random.normal(0, sigma, (neuron_count, self._input_shape))
            self.b = np.random.normal(0, sigma, neuron_count)
        elif init_distribution == 'uniform':
            l = self._l_dict[initialization](self._input_shape, self._output_shape)
            self.W = np.random.uniform(-l, l, (neuron_count, self._input_shape))
            self.b = np.random.uniform(-l, l, neuron_count)
        else:
            # for more methods
            raise Exception("WTF? Something went wrong")

    def __call__(self, x: np.ndarray, phase):
        assert x.shape[0] == self._input_shape
        self.x = x[:]
        self.a = self.W.dot(x) + self.b
        return self._activation_func(self.a)

    def is_trainable(self):
        return True

    def get_grad(self):
        assert self.dE_da is not None
        assert self.x is not None
        print(f"getgrad: {self.dE_da.shape} {self.x.shape}")
        return self.dE_da.reshape(-1, 1).dot(self.x.reshape(1, -1))

    def backward(self, dy):
        self.dE_da = self._activation_func.derivate(self.a) * dy
        self._previous_layer.backward(self.dE_da)

    def update_weights(self, update_func):
        dW = self.dE_da.reshape(-1, 1).dot(self.x.reshape(1, -1))
        self.W = update_func(self.W, dW)
        self.b = update_func(self.b, self.dE_da)

    def get_nrof_trainable_params(self):
        return self._input_shape * (self._output_shape + 1)

