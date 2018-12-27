import torch
from torch import nn
import helper
import torch.nn.functional as F
from torchvision import datasets, transforms
#
# class Network(nn.Module):
#     def __int__(self):
#         super.__init__()
#
#         self.hidden = nn.Linear(784, 256)
#         self.output = nn.Linear(256, 10)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.hidden(x)
#         x = self.signmoid(x)
#         x = self.output(x)
#         x = self.softmax(x)
#
#         return x
#

# using the concise functional declaration of the neural network

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        # Hidden layer with relu activation
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)

        return x


model = Network()
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Grab some data
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels)
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
# helper.view_classify(img.view(1, 28, 28), ps)

model = Network()
print(model)
# print(model.fc1.bias)