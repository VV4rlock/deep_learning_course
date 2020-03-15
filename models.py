from base import BaseModel, BaseActivationFunction
from layers import InputLayer, SoftmaxOutputLayer, FullyConnectedLayer
from activation_functions import ReLU, Linear
from collections.abc import Iterable

class MLP(BaseModel):
    types = ['classification', 'regression']
    def __init__(self, nrof_input, hidden_configure, activation_func, nrof_output, type='classification'):
        assert type in self.types
        assert isinstance(activation_func, BaseActivationFunction)
        assert isinstance(hidden_configure, Iterable)
        self.input_layer = InputLayer(nrof_input)
        self.hiddden_layers = []
        prev = self.input_layer
        for configure in hidden_configure:
            prev = FullyConnectedLayer(configure, prev, activation_func)
            self.hiddden_layers.append(prev)
        if type == self.types[0]:
            prev = FullyConnectedLayer(nrof_output, prev, activation_func)
            self.hiddden_layers.append(prev)
            self.out = SoftmaxOutputLayer(prev)
        elif type == self.types[1]:
            self.out = FullyConnectedLayer(nrof_output, prev, Linear)

    def __call__(self, input):
        res = self.input_layer(input, 'forward')
        for layer in self.hiddden_layers:
            res = layer(res, 'forward')
        res = self.out(res, 'forward')
        return res

    def train(self, dataloader, optimiser):
        generator = dataloader.batch_generator()
        batch = generator.__next__()
        for _ in range(100):
            for image, label in batch:
                input = image.reshape(28*28)
                self(input)
                self.out.backward(label)




