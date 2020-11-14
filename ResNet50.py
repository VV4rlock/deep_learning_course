import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import pickle
import cv2

import matplotlib.pyplot as plt
RESNET_C = True

class ResNetBlockBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, is_downsample=False):
        super(ResNetBlockBottleneck, self).__init__()
        assert out_channels % 4 == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        internal_channels = out_channels // 4
        self.shortcut_is_needed = in_channels == out_channels
        if is_downsample:
            stride = 2
            self.shortcut = nn.Sequential(
                nn.AvgPool2d((2, 2), stride=(stride, stride), padding=(0, 0)),
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                          bias=False),
                nn.BatchNorm2d(out_channels, affine=True, track_running_stats=False)
            )
        else:
            stride = 1
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                          bias=False),
                nn.BatchNorm2d(out_channels, affine=True, track_running_stats=False)
            )


        self.block = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),  #
            nn.BatchNorm2d(internal_channels, affine=True, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(internal_channels, internal_channels, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(internal_channels, affine=True, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(internal_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=False),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, class_count):
        super(ResNet, self).__init__()
        if RESNET_C:
            layers = [nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 112
                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))]
        else:
            layers = [nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)), #112
                      nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))] #56
        layers += [ResNetBlockBottleneck(64, 256, is_downsample=False)] #* 3
        layers += [ResNetBlockBottleneck(256, 512, is_downsample=True)] + [ResNetBlockBottleneck(512, 512, is_downsample=False)] * 3
        layers += [ResNetBlockBottleneck(512, 1024, is_downsample=True)] + [ResNetBlockBottleneck(1024, 1024, is_downsample=False)] #* 5
        layers += [ResNetBlockBottleneck(1024, 2048, is_downsample=True)] + [ResNetBlockBottleneck(2048, 2048, is_downsample=False)] #* 2
        layers += [nn.AvgPool2d((7, 7))]
        self.decoder = nn.Linear(2048, class_count)
        self.layers = nn.Sequential(
            *layers
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.decoder(x)

dummy = torch.ones((1, 3, 224, 224))
model = ResNet(class_count=12)

print(model)
print(model(dummy).shape)

loss_fn = nn.CrossEntropyLoss()
CUDA = torch.cuda.device_count()
device = 'cuda'
CIFAR_PATH = "/media/warlock/ssd/datasets/CIFAR"
DATASETS_DIR_PATH = "/media/warlock/ssd/datasets"
SOP_DEFINITION = "SOP_train_valid_split.pickle"

class SOP_classification(Dataset):
    def __init__(self, type='train', load_to_memory=True, transforms=None):
        assert type in ['test', 'train', 'valid']
        self.type = type
        self.load_to_memory = load_to_memory
        self.transforms = transforms
        self.samples = []
        self.labels = []
        toPIL = torchvision.transforms.ToPILImage()
        toTensor = torchvision.transforms.ToTensor()
        with open(SOP_DEFINITION, 'rb') as f:
            sop_dict = pickle.load(f)
        self.classes_names = sop_dict.keys()
        '''
            >>> data['bicycle_final']['train'].keys()
            dict_keys(['paths', 'product_labels', 'category_labels'])
        '''
        self.nrof_classes = len(self.classes_names)
        label_set = set()
        if self.load_to_memory:
            for class_index, _class in enumerate(self.classes_names):
                print(f"handling class {_class} ({class_index}/{self.nrof_classes}):")
                nrof_samples = len(sop_dict[_class][type]['paths'])
                for ind, name in enumerate(sop_dict[_class][type]['paths'][:200]):
                    print(f"\r\thandling image {ind}/{nrof_samples}: {name}...", end='')
                    filename = DATASETS_DIR_PATH + name
                    image = cv2.imread(filename)
                    self.samples.append(toPIL(image))
                    label = sop_dict[_class][type]['category_labels'][ind]
                    if label not in label_set:
                        label_set.add(label)
                    self.labels.append(label)
        print(f"\nCLASSES: {self.classes_names}: {label_set}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.transforms:
            sample = self.transforms(sample)
        return sample, self.labels[index]

train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(saturation=0.4, hue=0.4, brightness=0.4),
            transforms.ToTensor(),
            transforms.Normalize((123.68, 116.779, 103.939), (58.393, 57.12, 57.375))
        ]
    )

test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((123.68, 116.779, 103.939), (58.393, 57.12, 57.375))
        ]
    )
BS = 256
trainset = SOP_classification('train', load_to_memory=True, transforms=train_transform)
train_loader = DataLoader(trainset, batch_size=BS, shuffle=True)
testset = SOP_classification('valid', load_to_memory=True, transforms=test_transform)
test_loader = DataLoader(testset, batch_size=BS, shuffle=False)

def one_batch():
    mean_train_losses = []
    mean_test_losses = []
    test_acc_list = []
    BS = 20
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    epochs = 0
    epoch = 0
    images, labels = iter(test_loader).next()

    if CUDA:
        images = images.to(device)
        labels = labels.to(device)

        model.to(device)

    accuracy = 0
    while accuracy < 100:
        epoch += 1
        model.train()

        train_losses = []
        test_losses = []

        optimizer.zero_grad()

        outputs = model(images)

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
    mean_train_losses = []
    mean_test_losses = []
    test_acc_list = []
    train_acc_list = []
    BS = 256
    trainset = SOP_classification('train', load_to_memory=True, transforms=train_transform)
    train_loader = DataLoader(trainset, batch_size=BS, shuffle=True)
    testset = SOP_classification('valid', load_to_memory=True, transforms=test_transform)
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