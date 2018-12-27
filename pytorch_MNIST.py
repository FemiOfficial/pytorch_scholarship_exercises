import numpy as np
import torch

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# Define a transform to normalize the data

def activation(x):
    return 1/(1+torch.exp(-x))


transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

inputs = images.view(images.shape[0], -1)

W1 = torch.randn(784, 256)
b1 = torch.randn(256)

W2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = activation(torch.mm(W1 * inputs) + b1)
out = torch.mm()



