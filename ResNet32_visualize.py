import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from ResNet32_train import CIFAR_PATH, ResNet, ResNetBlock
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import cv2
import numpy as np


COUNT = 5

CUDA = torch.cuda.device_count()
device = 'cuda'
PATH = "resne32_t.pt"
model = torch.load(PATH, map_location=torch.device('cpu'))
model.eval()
loss_fn = torch.nn.CrossEntropyLoss()
transform = transforms.Compose(
        [
            #transforms.Pad((4, 4)),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32),
            transforms.ToTensor()
        ]
    )

def train_acc():

    BS=256
    model.eval()
    correct = 0
    total = 0
    trainset = torchvision.datasets.CIFAR10(CIFAR_PATH, train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=BS, shuffle=True)
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Train accuracy = {100 * correct / total}%")


def cam():
    BS = 1
    trainset = torchvision.datasets.CIFAR10(CIFAR_PATH, train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=BS, shuffle=True)

    # print(model)

    activation = None
    def hook(module, input, output):
        nonlocal activation
        activation = input[0].data.numpy()

    model.layers[16].register_forward_hook(hook)
    W = model.decoder.weight.data.numpy()
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            print(f"Predicted class: {predicted}, label: {labels}")


            pr = predicted.data.numpy()
            img = images.data.numpy()
            img_shape = img.shape[2:]

            cam = np.zeros(img.shape[2:], dtype=np.float64)
            for ind in range(activation.shape[1]):
                future = cv2.resize(activation[0, ind] * W[pr, ind], dsize=img_shape)
                cam += future
            # cam = cam - cam.min()
            plt.figure(figsize=(14, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(img[0].transpose(1, 2, 0))
            plt.imshow(cam, cmap='plasma', alpha=0.5)

            plt.subplot(1, 3, 2)
            plt.imshow(img[0].transpose(1, 2, 0))

            plt.subplot(1, 3, 3)
            plt.imshow(cam, cmap='plasma', alpha=0.5)
            plt.show()


def weights_vis():
    conv_w = model.layers[0].weight.data
    print(conv_w.shape)
    grid = make_grid(conv_w, nrow=4).numpy().transpose(1, 2, 0)
    print(grid.shape)
    plt.imshow(grid)
    plt.show()


def activation_vis():
    BS = 1
    trainset = torchvision.datasets.CIFAR10(CIFAR_PATH, train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=BS, shuffle=True)

    print(model)

    activation = None
    def hook(module, input, output):
        nonlocal activation
        activation = output.data

    model.layers[0].register_forward_hook(hook)
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):


            if i == COUNT:
                break
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            activation = activation.transpose(0, 1)
            grid = make_grid(activation, nrow=4).numpy().transpose((1,2,0))
            plt.imshow(grid, cmap='plasma')

            plt.show()


if __name__ == "__main__":
    # torch.save(model, PATH)
    # print(f"Model was dumped to {PATH}")
    #
    #cam()
    weights_vis()
    activation_vis()
    cam()





