import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
import json

# get the data

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# defining the transforms

train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.ToTensor(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

validation_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.CenterCrop(224),
                                           transforms.Resize(256),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testing_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.CenterCrop(224),
                                        transforms.Resize(256),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training_data = datasets.ImageFolder(train_dir, train_transform)
validation_dataset = datasets.ImageFolder(valid_dir, validation_transform)
testing_data = datasets.ImageFolder(test_dir, testing_transform)

# using data loaders
train_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=32)

# Label Mapping - located in a json file. Load in

with open('flower_to_name.json', 'r') as f:
    flower_to_name = json.load(f)

# Building the classifier model.

model = models.vgg16(pretrained=True)  # transfer learning, using a pretrained model
print(model)

# because the weights and hyper parameters of the model are already trained, we do not want to backpropogate through them
# and no need to do gradient descent, therefore we freeze the model parameters

for parameter in model.parameters():
    parameter.requires_grad = False

# above is the CNN part of the network which identifies features
# Build a custom feed forward fully connected classifier nerual network

classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                        ('activation1', nn.ReLU()),
                                        ('dropout later', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(in_features=5000, out_features=102)),
                                        ('output', nn.LogSoftmax(dim=1))]))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(25088, 5000)
        self.activation1 = nn.ReLU()
        nn.Dropout(p=0.5)  # to prevent overfitting, 50% chance a node "drops out"
        self.fc2 = nn.Linear(5000, 102)

    def forward(self, image):
        flattened = image.view(-1, 25088)
        output_of_input = self.fc1(flattened)
        activation_of_layer2 = nn.ReLU(output_of_input)
        output = self.fc2(activation_of_layer2)  # no activation at output or input layers.

        return F.softmax(output)  # unnesecary because cross entropy loss automatically applies it.


model.classifier = classifier

# defining criterion (loss functions) and optimizer (gradient descent)
# optimizer: Adam - replacement for SGD, better for noisy problems and sparse gradients

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# validation function
def validation(model, validation_loader, criterion):
    val_loss = 0
    accuracy = 0
    correct = 0
    total = 0

    for images, labels in iter(validation_loader):

        output = model.forward(images)
        probabilities = torch.exp(output) # because log max, need to take the exponential of results.

        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return accuracy


# training the custom classifier.
