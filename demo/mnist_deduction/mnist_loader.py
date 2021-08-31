import torch
import torchvision
from matplotlib import pyplot as plt


data = torchvision.datasets.MNIST('.', train=True, download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
         (0.1307,), (0.3081,))
    ])
)

print(data[0][0].shape)
