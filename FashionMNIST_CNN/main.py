# general machine learning pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# tensorboard to look at training data and compare results

# handle images
import torchvision
import torchvision.transforms as transforms

# import modules to build RunBuilder and RunManager helper classes
from collections import OrderedDict
from collections import namedtuple
from itertools import product

# libraries to display pictures
import matplotlib.pyplot as plt

# Loading in the data
transform = transforms.Compose([transforms.ToTensor()])

mnist_train = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                download=True,
                                                train=True,
                                                transform=transforms.ToTensor())

mnist_test = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                               download=True,
                                               train=False,
                                               transform=transforms.ToTensor())

mnist_test_images = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                      download=True,
                                                      train=False)

classes = {
    "0": "T-shirt/top",
    "1": "Trouser",
    "2": "Pullover",
    "3": "Dress",
    "4": "Coat",
    "5": "Sandal",
    "6": "Shirt",
    "7": "Sneaker",
    "8": "Bag",
    "9": "Ankle Boot"
}


# creating the CNN - screenshot of its architecture is in repo.
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN portion of the network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        # Fully connected layers
        self.fc1 = nn.Linear(12 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)

    def forward(self, x):
        # conv1 layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # conv2 layer
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # fully connected layers - first flatten data
        flattened = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1(flattened))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x  # don't use softmax because Cross entropy activation will be used later.


# Store hyperparameters inside a ordered dictionary

params = OrderedDict(
    learning_rate=[.01, .001],
    batch_size=[int(100), int(1000)],
    shuffle=[True, False]
)


def train(model, train_data, learning_rate, batch_size, num_epochs):
    criterion = nn.CrossEntropyLoss()  # loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch #", epoch, "Loss: ", loss.item())


# Get accuracy - helper function to determine the accuracy of the model
def get_accuracy(model, test_data):
    n_correct = 0
    n_total = 0
    batch_size = 32

    loader = DataLoader(test_data, batch_size=100)

    predictions = []
    answers = []

    for i, (imgs, labels) in enumerate(loader):
        output = model(imgs)
        _, predicted = torch.max(output, 1)

        for i in range(int(params.get("batch_size")[0])):
            answers.append(classes.get(str(labels[i].sum().item())))
            predictions.append(classes.get(str(predicted[i].sum().item())))

    for i in range(len(answers)):
        if answers[i] == predictions[i]:
            n_correct += 1
        n_total += 1

    accuracy = float(n_correct / n_total * 100)
    # print(f"Answers: {answers}")
    # print(f"Prediction: {predictions}")
    print(f"Number correct: {n_correct}, Accuracy: {accuracy}")

    return accuracy



model = MNISTClassifier()
train(model, mnist_train, learning_rate=params.get("learning_rate")[0], batch_size=params.get("batch_size")[0],
      num_epochs=10)

accuracy = get_accuracy(model, mnist_test)
