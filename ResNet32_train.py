import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

CIFAR_PATH = "/media/warlock/ssd/datasets/CIFAR"

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shortcut_is_needed = in_channels == out_channels
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
            stride = 1
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2), bias=False),
                nn.BatchNorm2d(out_channels, affine=True, track_running_stats=False)
                                          )
            stride = 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=False)
        )
        self.relu = nn.ReLU()


    def forward(self, x):
        residual = self.shortcut(x)
        return self.relu(self.block(x) + residual)


class ResNet(nn.Module):
    def __init__(self, N=5):
        super(ResNet, self).__init__()
        layers = [nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
        layers += [ResNetBlock(16, 16)] * N
        layers += [ResNetBlock(16, 32)] + [ResNetBlock(32, 32)] * (N - 1)
        layers += [ResNetBlock(32, 64)] + [ResNetBlock(64, 64)] * (N - 1)
        layers += [nn.AvgPool2d((8, 8))]
        self.decoder = nn.Linear(64, 10)
        self.layers = nn.Sequential(
            *layers
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.decoder(x)

dummy = torch.ones((1, 3, 32, 32))
transform = transforms.Compose(
        [
            transforms.Pad((4, 4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor()
        ]
    )

#print(model)
#print(model(dummy).shape)

loss_fn = nn.CrossEntropyLoss()
CUDA = torch.cuda.device_count()
device = 'cuda'

def one_batch():
    model = ResNet(N=5)
    mean_train_losses = []
    mean_test_losses = []
    test_acc_list = []
    BS = 200
    trainset = torchvision.datasets.CIFAR10(CIFAR_PATH, train=True, download=False, transform=transform)
    dl = DataLoader(trainset, batch_size=BS, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    epochs = 0
    epoch = 0
    images, labels = iter(dl).next()

    images = images.to(device)
    labels = labels.to(device)

    model.to(device)

    print(images.element_size())
    print(labels.element_size())

    accuracy = 0
    while accuracy < 100:
        epoch += 1
        model.train()

        train_losses = []
        test_losses = []

        optimizer.zero_grad()

        outputs = model(images)
        print("outpus size", outputs.element_size())
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        correct = 0
        total = 0

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        mean_train_losses.append(np.mean(train_losses))

        accuracy = 100 * correct / total
        test_acc_list.append(accuracy)
        print('epoch : {}, train loss : {:.4f}, test loss : {:.4f}, test acc : {:.2f}%' \
              .format(epoch + 1, np.mean(train_losses), np.mean(test_losses), accuracy))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    ax1.plot(mean_train_losses, label='train')
    # ax1.plot(mean_test_losses, label='test')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')

    ax2.plot(test_acc_list, label='test acc')
    ax2.legend()
    plt.show()


def train():
    model = ResNet(N=5)
    mean_train_losses = []
    mean_test_losses = []
    test_acc_list = []
    train_acc_list = []
    BS = 256
    trainset = torchvision.datasets.CIFAR10(CIFAR_PATH, train=True, download=False, transform=transform)
    train_loader = DataLoader(trainset, batch_size=BS, shuffle=True)
    testset = torchvision.datasets.CIFAR10(CIFAR_PATH, train=False, download=False, transform=transform)
    test_loader = DataLoader(testset, batch_size=BS, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    epochs = 200
    best_acc = 0
    PATH = "drive/My Drive/resne32.pt"
    # torch.save(model, PATH)
    # print(f"Model was dumped to {PATH}")

    # model = torch.load(PATH)
    # model.eval()
    print("train started")
    if CUDA:
        device = 'cuda'
        model.to(device)
    for epoch in range(epochs):

        model.train()

        correct = 0
        total = 0
        train_losses = []
        test_losses = []
        for i, (images, labels) in enumerate(train_loader):

            if CUDA:
                images = images.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        t_accuracy = 100 * correct / total
        train_acc_list.append(t_accuracy)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                if CUDA:
                    images = images.to(device)
                    labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                test_losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        mean_train_losses.append(np.mean(train_losses))
        mean_test_losses.append(np.mean(test_losses))

        accuracy = 100 * correct / total
        test_acc_list.append(accuracy)
        if accuracy > best_acc:
            torch.save(model, PATH)
            best_acc = accuracy
            print(f"Model was dumped to {PATH}")
        print('epoch : {}, train loss : {:.4f}, test loss : {:.4f}, train acc : {:.2f}%, test acc : {:.2f}%' \
              .format(epoch + 1, np.mean(train_losses), np.mean(test_losses), t_accuracy, accuracy))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    ax1.plot(mean_train_losses, label='train')
    ax1.plot(mean_test_losses, label='test')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')

    ax2.plot(train_acc_list, label='train acc')
    ax2.plot(test_acc_list, label='test acc')
    ax2.legend()
    plt.show()

if __name__=="__main__":
    one_batch()
    #train()
