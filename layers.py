from base import BaseLayerClass, BaseActivationFunction
from config import *


class InputLayer(BaseLayerClass):
    def __init__(self, shape):
        self._input_shape = shape
        self._output_shape = shape
        self._previous_layer = None

    def __call__(self, x):
        return x.reshape(self._output_shape)

    def is_trainable(self):
        return False

    def is_input_layer(self):
        return True

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

    def __call__(self, x):
        e = np.exp(x - x.max())
        self.y = e / e.sum()
        return self.y

    def is_trainable(self):
        return False

    def backward(self, dz_dl):
        return dz_dl

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

    def __call__(self, x):
        self.y = x[:]
        return x

    def is_trainable(self):
        return False

    def backward(self, dz_dl):
        return dz_dl

    def update_weights(self, update_func):
        logger.warn("Output layer is not trainable")

    def get_grad(self):
        logger.warn("Output layer is not trainable")

    def get_nrof_trainable_params(self):
        return 0


class FullyConnectedLayer(BaseLayerClass):
    _distributions = ['uniform', 'normal']
    _sigma_dict = {                                     # TASK: сделать 1 словарь, батчи
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
        self.neuron_count = neuron_count
        self._output_shape = neuron_count
        self._previous_layer = previous_layer
        self.a = None
        self.dE_da = None
        self.x = None
        self.b = np.zeros(neuron_count)
        if init_distribution == 'normal':
            sigma = self._sigma_dict[initialization](self._input_shape, self._output_shape)
            self.W = np.random.normal(0, sigma, (neuron_count, self._input_shape))
        elif init_distribution == 'uniform':
            l = self._l_dict[initialization](self._input_shape, self._output_shape)
            self.W = np.random.uniform(-l, l, (neuron_count, self._input_shape))
        else:
            # for more methods
            raise Exception("WTF? Something went wrong")

    def __call__(self, x: np.ndarray):
        assert x.shape[0] == self._input_shape
        self.x = x[:]
        self.a = self.W.dot(x) + self.b
        return self._activation_func(self.a)

    def is_trainable(self):
        return True

    def get_grad(self):
        assert self.dE_da is not None
        assert self.x is not None
        return (self.dE_da.reshape(-1, 1).dot(self.x.reshape(1, -1)), self.dE_da)

    def backward(self, dy):
        self.dE_da = self._activation_func.derivate(self.a) * dy
        return np.dot(self.W.T, self.dE_da)

    def update_weights(self, param_delta):
        assert len(param_delta) == 2
        self.W += param_delta[0]
        self.b += param_delta[1]

    def get_nrof_trainable_params(self):
        return self.W.size + self.b.size


class ConvLayer(BaseLayerClass):
    initialization_funcs = {'he_normal': lambda n, m, shape: np.random.normal(0, (2 / n) ** 0.5, shape),
                            'xavier_normal': lambda n, m, shape: np.random.normal(0, (1 / n) ** 0.5, shape),
                            'glorot_normal': lambda n, m, shape: np.random.normal(0, (1 / (n + m)) ** 0.5, shape),
                            'he_uniform': lambda n, m, shape: np.random.uniform(-(6 / n) ** 0.5, (6 / n) ** 0.5, shape),
                            'xavier_uniform': lambda n, m, shape: np.random.uniform(-(3 / n) ** 0.5, (3 / n) ** 0.5, shape),
                            'glorot_uniform': lambda n, m, shape: np.random.uniform(-(6 / (n + m)) ** 0.5, (6 / (n + m)) ** 0.5, shape)
                            }

    def __init__(self, input_shape, n_filters, k_height, k_width, activation_func, pad=0, stride=1, initialization='he_normal', prev_layer=None):
        assert len(input_shape) == 4
        assert initialization in ConvLayer.initialization_funcs
        m = n_filters
        self.n_filters = n_filters
        self.batch_size, self.x_depth, self.H, self.W = input_shape
        self.pad = pad
        if pad != 0:
            self.H, self.W = self.H + 2*pad, self.W + 2*pad
        n = k_height * k_width * self.x_depth
        self.out_width = (self.W - k_width) // stride + 1
        self.out_height = (self.H - k_height) // stride + 1
        self._output_shape = (self.batch_size, self.x_depth, self.out_height, self.out_width)
        self.k_height = k_height
        self.k_width = k_width
        self.prev_layer = prev_layer
        self.kernels = ConvLayer.initialization_funcs[initialization](n, m, (n_filters, k_width * k_height * self.x_depth))
        self.bias = np.zeros(n_filters, dtype=np.float32).reshape(-1, 1)
        self.k, self.i, self.j = ConvLayer.get_indices((self.batch_size, self.x_depth, self.H, self.W), k_height, k_width, stride=stride)
        self.activation_func = activation_func

    @staticmethod
    def get_indices(one_image_shape, k_height, k_width, stride=1):
        '''
        лучше принимать x.shape вместо x и хранить внутри слоя все индексы
            непонятно зачем использовать паддинг, если его можно применить заранее
        pad уже не нужен, т к применен.
        Если изображение имеет не кратное число пикселей, то часть изображения свертка не увидит?
        :param one_image_shape: размеры входного тензора (batchsize, channel_count, H, W)
        :param k_width: размеры ядра сверткм
        :param k_height:
        :param stride:
        :return:
        '''
        assert len(one_image_shape) == 4
        batch_size, depth, H, W = one_image_shape
        out_width, out_height = (W - k_width) // stride + 1, (H - k_height) // stride + 1
        # stride forgotten
        k = np.arange(depth).repeat(out_height * out_width).reshape((-1, 1))
        i = np.tile(
            np.tile(np.tile(np.arange(k_width).repeat(k_height), (out_height, 1)), (out_width, 1)) + stride * np.arange(
                out_width).repeat(out_height).reshape(-1, 1), (depth, 1))
        j = np.tile(np.tile(
            np.expand_dims(np.tile(np.arange(k_height), k_width), 0).repeat(out_height, 0) + stride * np.arange(
                out_height).reshape(-1, 1), (out_width, 1)), (depth, 1))
        return k, i, j  # в презентации ошибка, надо брать индексы не как x[:,k,i,j] а как x[:,k,j,i]?

    @staticmethod
    def zeroPad(x, pad):
        assert x.ndim == 4
        return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    def _set_kernels(self, kernels):
        assert kernels.ndim == 4
        self.n_filters, self.x_depth, self.k_height, self.k_width = kernels.shape
        self.kernels = kernels.transpose(0, 1, 3, 2).reshape((self.n_filters, self.k_width * self.k_height * self.x_depth))

    def _set_bias(self, bias):
        self.bias = bias.reshape(self.bias.shape[0], 1)

    def __call__(self, x):
        if self.pad:
            x = ConvLayer.zeroPad(x, self.pad)

        self.a = (np.dot(self.kernels, x[:, self.k, self.j, self.i]
                .reshape((self.batch_size, self.x_depth, self.out_height*self.out_width, self.k_height*self.k_width))
                .transpose(1, 3, 2, 0)
                .reshape((self.k_height * self.k_width * self.x_depth, -1))) + self.bias) \
            .reshape((self.n_filters, self.out_width, self.out_height, self.batch_size)).transpose((3, 0, 2, 1))
        return self.activation_func(self.a)


class MaxPooling(BaseLayerClass):
    def __init__(self, input_shape, k_height, k_width, stride=1): # pad is needed?
        self.k_height = k_height
        self.k_width = k_width
        self.stride = stride
        self.batch_size, self.x_depth, self.H, self.W = input_shape
        self.out_width = (self.W - k_width) // stride + 1
        self.out_height = (self.H - k_height) // stride + 1
        self._output_shape = (self.batch_size, self.x_depth, self.out_height, self.out_width)
        self.k, self.i, self.j = ConvLayer.get_indices((self.batch_size, self.x_depth, self.H, self.W), k_height,
                                                       k_width, stride=stride)

    def __call__(self, x):
        return np.max(x[:, self.k, self.j, self.i], axis=2).reshape((self.batch_size, self.x_depth, self.out_width, self.out_height)).transpose(0, 1, 3, 2)


class AveragePooling(MaxPooling):
    def __call__(self, x):
        return np.sum(x[:, self.k, self.j, self.i], axis=2).reshape((self.batch_size, self.x_depth, self.out_width, self.out_height)).transpose(0, 1, 3, 2) / (self.k_width * self.k_height)
