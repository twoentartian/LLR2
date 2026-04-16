import torch
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np

def weights_init_xavier(module):
    if isinstance(module, nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        torch.nn.init.constant_(module.bias, 0)

""" LeNet """
class LeNet4(nn.Module):
    def __init__(self):
        super(LeNet4, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)  # matches parameters in lenet_train.prototxt
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)  # Flatten from conv2 output size (50, 4, 4)
        self.fc2 = nn.Linear(500, 10)  # 10 classes for MNIST
        self.apply(weights_init_xavier)

    def forward(self, x):
        x = self.conv1(x)
        x = nnF.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nnF.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)  # Flatten from 2D to 1D
        x = self.fc1(x)
        x = nnF.relu(x)
        x = self.fc2(x)
        return x


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.apply(weights_init_xavier)

    def forward(self, x):
        x = nnF.max_pool2d(nnF.relu(self.conv1(x)), (2, 2))
        x = nnF.max_pool2d(nnF.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = nnF.relu(self.fc1(x))
        x = nnF.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        return np.prod(size)


class LeNet5LargeFc(nn.Module):
    def __init__(self):
        super(LeNet5LargeFc, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 360)
        self.fc2 = nn.Linear(360, 256)
        self.fc3 = nn.Linear(256, 10)
        self.apply(weights_init_xavier)

    def forward(self, x):
        x = nnF.max_pool2d(nnF.relu(self.conv1(x)), (2, 2))
        x = nnF.max_pool2d(nnF.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = nnF.relu(self.fc1(x))
        x = nnF.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        return np.prod(size)

""" LeNet helper functions """
def lenet4():
    return LeNet4()

def lenet5(large_fc=False):
    if large_fc:
        return LeNet5LargeFc()
    else:
        return LeNet5()