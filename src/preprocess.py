import torch
from torch.utils.data import DataLoader

def preprocess(train, test):

    train_loader = DataLoader(train, batch_size=256, shuffle=True)
    test_loader = DataLoader(test, batch_size=256, shuffle=False)

    return train_loader, test_loader


# 🔹 Flatten function (784 vector)
def flatten(images):
    return images.view(images.size(0), -1)


# 🔹 Add noise
def add_noise(images, noise_factor=0.5):

    noisy = images + noise_factor * torch.randn_like(images)
    noisy = torch.clamp(noisy, 0., 1.)

    return noisy