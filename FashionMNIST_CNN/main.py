# general machine learning pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# tensorboard to look at training data and compare results

# handle images
import torchvision
import torchvision.transforms as transforms

# Loading in the data
transform = transforms.Compose(transforms.ToTensor())
mnist_train = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                download=True,
                                                train=True,
                                                transform=transform)


# creating the CNN - screenshot of its architecture is in repo.
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN portion of the network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        # Fully connected layers
        self.fc1 = nn.Linear(12*4*4, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)

    def forward(self,x):
        #conv1 layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # conv2 layer
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # fully connected layers - first flatten data
        flattened = x.view(-1,12*4*4)
        x = F.relu(self.fc1(flattened))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x # don't use softmax because Cross entropy activation will be used later.
