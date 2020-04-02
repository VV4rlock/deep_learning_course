from base import BaseModel, BaseActivationFunction
from layers import InputLayer, SoftmaxOutputLayer, FullyConnectedLayer, SimpleOutputLayer
from activation_functions import ReLU, Linear
from collections.abc import Iterable
import pickle
import matplotlib.pyplot as plt
import numpy as np

class MLP(BaseModel):
    types = ['classification', 'regression']

    def __init__(self, nrof_input, hidden_configure, activation_func, nrof_output, type='classification'):
        assert type in self.types
        assert isinstance(activation_func, BaseActivationFunction)
        assert isinstance(hidden_configure, Iterable)
        self.param_count = 0
        self.input_layer = InputLayer(nrof_input)
        self.hiddden_layers = []
        self.nrof_output = nrof_output
        prev = self.input_layer
        self.config = hidden_configure
        for configure in hidden_configure:
            prev = FullyConnectedLayer(configure, prev, activation_func)
            self.param_count += prev.get_nrof_trainable_params()
            self.hiddden_layers.append(prev)
        prev = FullyConnectedLayer(nrof_output, prev, Linear())
        self.param_count += prev.get_nrof_trainable_params()
        self.hiddden_layers.append(prev)
        if type == self.types[0]:
            self.out = SoftmaxOutputLayer(prev)
        elif type == self.types[1]:
            self.out = SimpleOutputLayer(prev)

    def get_nrof_trainable_params(self):
        return self.param_count

    def __call__(self, input):
        res = self.input_layer(input)
        for layer in self.hiddden_layers:
            res = layer(res)
        res = self.out(res)
        return res

    def validate(self, generator):
        hit, count, loss = 0, 0, 0
        for input, one_hot_vector in generator:
            res = self(input)
            count += one_hot_vector
            loss -= np.log(res[one_hot_vector.argmax()])
            if one_hot_vector.argmax() == res.argmax():
                hit += one_hot_vector
        return loss, hit, count




