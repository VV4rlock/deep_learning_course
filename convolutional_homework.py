import numpy as np
import torch
import torch.nn as nn

from activation_functions import ReLU, Linear
from layers import ConvLayer, MaxPooling, AveragePooling
from base import BaseLayerClass
from tests import indices_tests, cont_tests


def indices_test():
    print("Testing get_indices...")
    for name in indices_tests:
        inp, out = indices_tests[name]
        k, i, j = ConvLayer.get_indices(*inp)
        if np.all(k == out[0]) and np.all(i == out[1]) and np.all(j == out[2]):
            print(f"get_indices test {name}: OK")
        else:
            print(f"get_indices test {name}: Failed. Expected:")
            print(k, i, j, sep='\n')


def conv_test():
    print("Testing conv layers")
    eps = 0.01
    gen_sample = lambda size, channels : np.stack([np.random.uniform(-10, 10, (size, size)) for i in range(channels)], axis=0)
    for name in cont_tests:
        batch_size, im_s, channels, n_filters, k_size, stride, padding = cont_tests[name]

        batch = np.stack([gen_sample(im_s, channels) for j in range(batch_size)], axis=0)
        kernel = np.random.uniform(-3.0, 3.0, (n_filters, channels, k_size[0], k_size[1]))
        bias = np.random.uniform(-3.0, 3.0, n_filters)

        my_conv = ConvLayer(batch.shape, n_filters, k_size[0], k_size[1], Linear(), pad=padding, stride=stride)
        my_conv._set_kernels(kernel)
        my_conv._set_bias(bias)
        my_max = MaxPooling(batch.shape, k_size[0], k_size[1], stride=stride)
        my_avg = AveragePooling(batch.shape, k_size[0], k_size[1], stride=stride)

        torch_conv = nn.Conv2d(channels, n_filters, k_size, stride=stride, padding=padding)
        torch_conv.weight = nn.Parameter(torch.tensor(kernel).float())
        torch_conv.bias = nn.Parameter(torch.tensor(bias).float())
        torch_max = nn.MaxPool2d(k_size, stride=stride)
        torch_avg = nn.AvgPool2d(k_size, stride=stride)

        print(f"test {name}:")
        print(f"\tconv:\t{np.all(my_conv(batch) - torch_conv(torch.tensor(batch).float()).detach().numpy() < eps)}")
        print(f"\tmax:\t{np.all(my_max(batch) - torch_max(torch.tensor(batch).float()).detach().numpy() < eps)}")
        print(f"\tavg:\t{np.all(my_avg(batch) - torch_avg(torch.tensor(batch).float()).detach().numpy() < eps)}")
    print("Success")


if __name__ == "__main__":
    indices_test()
    conv_test()