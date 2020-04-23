import numpy as np
from activation_functions import ReLU, Linear
from layers import ConvLayer, MaxPooling, AveragePooling
import torch
import torch.nn as nn

batch_size, im_s, channels, n_filters, k_size, stride, padding = (2, 3, 2, 2, (2, 2), 1, 0)
out_width, out_height = (im_s - k_size[0]) // stride + 1, (im_s - k_size[1]) // stride + 1
batch = np.stack([np.stack([np.arange(im_s ** 2).reshape((im_s, im_s)) + i * im_s ** 2 for i in range(channels)],
                               axis=0) + im_s ** 2 * channels * j for j in range(batch_size)], axis=0)

k, i, j = ConvLayer.get_indices(batch.shape[1:], k_size[0], k_size[1], stride)

gen_sample = lambda size, channels: np.stack([np.random.uniform(-10, 10, (size, size)) for i in range(channels)],
                                             axis=0)
batch = np.stack([gen_sample(im_s, channels) for j in range(batch_size)], axis=0)
kernel = np.random.uniform(-3.0, 3.0, (n_filters, channels, k_size[0], k_size[1]))
bias = np.random.uniform(-3.0, 3.0, n_filters)

torch_conv = nn.Conv2d(channels, n_filters, k_size, stride=stride, padding=padding)
torch_conv.weight = nn.Parameter(torch.tensor(kernel).float())
torch_conv.bias = nn.Parameter(torch.tensor(bias).float())
torch_max = nn.MaxPool2d(k_size, stride=stride)
torch_avg = nn.AvgPool2d(k_size, stride=stride)



#dz[:,k,j,i] += matr.reshape((batch_size,k_size[0]*k_size[1]*channels,out_width*out_height))