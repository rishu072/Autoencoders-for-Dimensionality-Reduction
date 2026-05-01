import torch
from torchvision import datasets, transforms

def load_data():

    transform = transforms.ToTensor()

    train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return train, test