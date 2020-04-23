from abc import *
import numpy as np
import pickle
import matplotlib.pyplot as plt


class BaseActivationFunction:
    @abstractmethod
    def __call__(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivate(self, input: np.ndarray) -> np.ndarray:
        pass


class BaseLayerClass:
    @abstractmethod
    def __call__(self, x):
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

    def is_input_layer(self):
        return False

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

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.learning_rate

    def minimise(self, batch, labels):
        batch_size = batch.shape[0]
        net_params = []
        loss, hit, count = 0, 0, 0
        label_bias = self.label_smoothing / self.net.nrof_output
        label_mul = (1 - self.label_smoothing)

        #for input, t in batch:
        res = self.net(batch)
        loss += self.loss_func(res, labels)
        count += labels.sum(axis=0)
        for i in range(batch.shape[0]):
            if labels[i].argmax() == res[i].argmax():
                hit += labels[i]

        if self.label_smoothing:
            labels = labels * label_mul + label_bias

        dl_dz = res - labels  # add loss_func_derivate

        temp_layer_grads = []
        prev_layer = self.net.get_last_layer()
        while prev_layer.get_previous_layer():
            dl_dz = prev_layer.backward(dl_dz)
            if prev_layer.is_trainable():
                temp_layer_grads.append(prev_layer.get_grad())
            prev_layer = prev_layer.get_previous_layer()

        dl_dz = prev_layer.backward(dl_dz)
        if prev_layer.is_trainable():
            temp_layer_grads.append(prev_layer.get_grad())
        #net_params.append(temp_layer_grads)

        p = self.update_layers_rule(temp_layer_grads)

        prev_layer = self.net.get_last_layer()
        layer_index = 0
        while not prev_layer.is_input_layer():
            if prev_layer.is_trainable():
                prev_layer.update_weights(p[layer_index])
                layer_index += 1
            prev_layer = prev_layer.get_previous_layer()
        if isinstance(hit, int):
            hit = count * 0
        return loss, hit, count


class BaseModel:
    @abstractmethod
    def __call__(self, input):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    def train(self, batch_generator, optimizer):
        error_rate, cross_entripy = [], []
        loss, hit, count = 0, 0, 0
        for batch, labels in batch_generator:
            loss, hit, count = optimizer.minimise(batch, labels)
        error_rate.append(1 - hit.sum()/count.sum())
        cross_entripy.append(loss)
        return error_rate, cross_entripy

    def dump_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Model was dumped to {filename}")

    def get_last_layer(self):
        return self.out

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model was loaded from {filename}")
        return model

