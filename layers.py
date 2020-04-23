from base import BaseLayerClass, BaseActivationFunction
from config import *


class InputLayer(BaseLayerClass):
    def __init__(self, shape):
        self._input_shape = shape
        self._output_shape = shape
        self._previous_layer = None

    def __call__(self, x):
        '''input is (batchsize, ...)'''
        return x.reshape((-1, self._output_shape))

    def is_trainable(self):
        return False

    def is_input_layer(self):
        return True

    def backward(self, dy):
        return dy

    def update_weights(self, update_func):
        logger.warn("Input layer is not trainable")

    def get_grad(self):
        logger.warn("Input layer is not trainable")

    def get_nrof_trainable_params(self):
        return 0


class ReshapeLayer(BaseLayerClass):
    def __init__(self, input_shape, output_shape, prev_layer=None):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._previous_layer = prev_layer

    def __call__(self, x):
        '''input is (batchsize, ...)'''
        return x.reshape((-1, *self._output_shape))

    def is_trainable(self):
        return False

    def is_input_layer(self):
        return True

    def backward(self, dy):
        return dy.reshape(-1, *self._input_shape)

    def update_weights(self, update_func):
        logger.warn("Reshape layer is not trainable")

    def get_grad(self):
        logger.warn("Reshape layer is not trainable")

    def get_nrof_trainable_params(self):
        return 0



class SoftmaxOutputLayer(BaseLayerClass):
    def __init__(self, prev_layer):
        assert isinstance(prev_layer, BaseLayerClass)
        self._input_shape = prev_layer.get_output_shape()
        self._output_shape = self._input_shape
        self._previous_layer = prev_layer
        self.y = None

    def __call__(self, x):
        e = np.exp(x - x.max(axis=1).reshape(-1, 1))
        self.y = e / e.sum(axis=1).reshape(-1, 1)
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
    def __init__(self, prev_layer):
        assert isinstance(prev_layer, BaseLayerClass)
        self._input_shape = prev_layer.get_output_shape()
        self._output_shape = self._input_shape
        self._previous_layer = prev_layer
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
    initialization_funcs = {'he_normal': lambda n, m, shape: np.random.normal(0, (2 / n) ** 0.5, shape),
                            'xavier_normal': lambda n, m, shape: np.random.normal(0, (1 / n) ** 0.5, shape),
                            'glorot_normal': lambda n, m, shape: np.random.normal(0, (1 / (n + m)) ** 0.5, shape),
                            'he_uniform': lambda n, m, shape: np.random.uniform(-(6 / n) ** 0.5, (6 / n) ** 0.5, shape),
                            'xavier_uniform': lambda n, m, shape: np.random.uniform(-(3 / n) ** 0.5, (3 / n) ** 0.5,
                                                                                    shape),
                            'glorot_uniform': lambda n, m, shape: np.random.uniform(-(6 / (n + m)) ** 0.5,
                                                                                    (6 / (n + m)) ** 0.5, shape)
                            }

    def __init__(self, input_shape, neuron_count, activation_func, initialization='he_normal', prev_layer=None):
        assert initialization in self.initialization_funcs
        assert isinstance(prev_layer, BaseLayerClass)
        assert isinstance(activation_func, BaseActivationFunction)
        self._activation_func = activation_func
        self._input_shape = input_shape
        self.neuron_count = neuron_count
        self._output_shape = neuron_count
        self._previous_layer = prev_layer
        self.a = None
        self.dE_da = None
        self.x = None
        self.b = np.zeros(neuron_count)
        self.W = self.initialization_funcs[initialization](self._input_shape, self._output_shape, (self._input_shape, neuron_count))

    def __call__(self, x: np.ndarray):
        assert x.shape[1] == self._input_shape
        self.x = x[:]
        #self.a = self.W.dot(x) + self.b
        self.a = x.dot(self.W) + self.b
        return self._activation_func(self.a)

    def is_trainable(self):
        return True

    def get_grad(self):
        assert self.dE_da is not None
        assert self.x is not None

        return (np.matmul(np.expand_dims(self.dE_da, axis=2), np.expand_dims(self.x, axis=1)).sum(axis=0).T / self.bs, self.dE_da.sum(axis=0) / self.bs)

    def backward(self, dy):
        assert dy.shape == self.a.shape
        self.bs = dy.shape[0]
        self.dE_da = self._activation_func.derivate(self.a) * dy
        return np.dot(self.dE_da, self.W.T) #???

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

    def __init__(self, input_shape, n_filters, k_height, k_width, activation_func, pad=0, stride=1, initialization='he_normal', use_bias=True, prev_layer=None):
        assert len(input_shape) == 3 # without batchsize
        assert initialization in ConvLayer.initialization_funcs
        m = n_filters
        self.n_filters = n_filters
        self.x_depth, self.H, self.W = input_shape
        self.pad = pad
        if pad != 0:
            self.H, self.W = self.H + 2*pad, self.W + 2*pad
        n = k_height * k_width * self.x_depth
        self.out_width = (self.W - k_width) // stride + 1
        self.out_height = (self.H - k_height) // stride + 1
        self.out_size = self.out_height * self.out_width
        self._output_shape = (self.n_filters, self.out_height, self.out_width)
        self.k_height = k_height
        self.k_width = k_width
        self.kern_size = k_height * k_width
        self._previous_layer = prev_layer
        self.kernels = ConvLayer.initialization_funcs[initialization](n, m, (n_filters, self.kern_size * self.x_depth))
        self.bias = np.zeros(n_filters, dtype=np.float32).reshape(-1, 1)
        self.k, self.i, self.j = ConvLayer.get_indices((self.x_depth, self.H, self.W), k_height, k_width, stride=stride)
        self.activation_func = activation_func
        self.use_bias = use_bias

    def get_nrof_trainable_params(self):
        return self.kernels.size + self.bias.size

    @staticmethod
    def get_indices(one_image_shape, k_height, k_width, stride=1):
        '''
        лучше принимать x.shape вместо x и хранить внутри слоя все индексы
            непонятно зачем использовать паддинг, если его можно применить заранее
        pad уже не нужен, т к применен.
        Если изображение имеет не кратное число пикселей, то часть изображения свертка не увидит?
        :param one_image_shape: размеры входного тензора (channel_count, H, W)
        :param k_width: размеры ядра сверткм
        :param k_height:
        :param stride:
        :return:
        '''
        assert len(one_image_shape) == 3 #without batchsize
        depth, H, W = one_image_shape
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
        return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    def _set_kernels(self, kernels):
        assert kernels.ndim == 4
        self.n_filters, self.x_depth, self.k_height, self.k_width = kernels.shape
        self.kern_size = self.k_height * self.k_width
        self.kernels = kernels.transpose(0, 1, 3, 2).reshape((self.n_filters, self.kern_size * self.x_depth))

    def _set_bias(self, bias):
        self.bias = bias.reshape(self.bias.shape[0], 1)

    def __call__(self, x):
        assert x.ndim == 4
        if self.pad:
            x = ConvLayer.zeroPad(x, self.pad)
        self.forward_shape = x.shape
        self.im2matr = x[:, self.k, self.j, self.i].reshape((-1, self.x_depth, self.out_size, self.kern_size))\
            .transpose(1, 3, 2, 0).reshape((self.k_height * self.k_width * self.x_depth, -1))

        self.a = (np.dot(self.kernels, self.im2matr) + self.bias) \
            .reshape((self.n_filters, self.out_width, self.out_height, -1)).transpose((3, 0, 2, 1))
        return self.activation_func(self.a)

    def backward(self, dz):
        da = (self.activation_func.derivate(self.a) * dz).transpose(1, 3, 2, 0).reshape(self.n_filters, -1)
        self.dw = np.dot(da, self.im2matr.T) / dz.shape[0]
        self.db = da.sum(axis=1) / dz.shape[0]
        matr = np.dot(self.kernels.T, da)\
            .reshape((self.x_depth, self.kern_size, self.out_size, -1))\
            .transpose(3, 0, 2, 1)
        out = np.zeros(self.forward_shape)
        t = out[:,self.k,self.j,self.i]
        np.add.at(out, (..., self.k, self.j, self.i), matr.reshape(-1, self.x_depth * self.out_size, self.kern_size))
        return out[:, :, self.pad:-self.pad, self.pad:-self.pad]

    def get_grad(self):
        return (self.dw, self.db)

    def update_weights(self, param_delta):
        assert len(param_delta) == 2
        self.kernels += param_delta[0]
        if self.use_bias:
            self.bias += param_delta[1]

    def is_trainable(self):
        return True


class MaxPooling(BaseLayerClass):
    def __init__(self, input_shape, k_height, k_width, stride=1, prev_layer=None): # pad is needed?
        self.k_height = k_height
        self.k_width = k_width
        self.stride = stride
        self.x_depth, self.H, self.W = input_shape
        self.out_width = (self.W - k_width) // stride + 1
        self.out_height = (self.H - k_height) // stride + 1
        self._output_shape = (self.x_depth, self.out_height, self.out_width)
        self.k, self.i, self.j = ConvLayer.get_indices((self.x_depth, self.H, self.W), k_height,
                                                       k_width, stride=stride)
        self._previous_layer = prev_layer

    def __call__(self, x):
        self.inp_shape = x.shape
        self.x = x[:, self.k, self.j, self.i]
        self.m = np.max(self.x, axis=2)
        return self.m.reshape((-1, self.x_depth, self.out_width, self.out_height)).transpose(0, 1, 3, 2)

    def backward(self, dy):
        out = np.zeros(self.inp_shape)
        dL = dy.transpose(0, 1, 3, 2).reshape(-1, self.out_width*self.out_height*self.x_depth, 1)
        np.add.at(out, (..., self.k, self.j, self.i), np.where(self.x == self.m, dL, 0))
        return out

    def is_trainable(self):
        return False

    def get_nrof_trainable_params(self):
        return 0


class AveragePooling(MaxPooling):
    def __call__(self, x):
        self.inp_shape = x.shape
        self.x = x[:, self.k, self.j, self.i]
        return np.sum(self.x, axis=2).reshape((-1, self.x_depth, self.out_width, self.out_height)).transpose(0, 1, 3, 2) / (self.k_width * self.k_height)

    def backward(self, dy):
        out = np.zeros(self.inp_shape)
        dL = dy.transpose(0, 1, 3, 2).reshape(-1, self.out_width * self.out_height * self.x_depth, 1)
        np.add.at(out, (..., self.k, self.j, self.i),
                  np.ones((self.inp_shape[0], self.out_height * self.out_width * self.x_depth, self.k_height * self.k_width))
                  * dL / (self.k_width * self.k_height))
        return out
