import torch
import torch.nn as nn
import torch.optim as optim
from src.preprocess import flatten

def train_model(model, train_loader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for epoch in range(5):   # start small (later 50)

        running_loss = 0

        for images, _ in train_loader:

            # flatten images
            images = flatten(images)

            # forward
            outputs = model(images)

            loss = criterion(outputs, images)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        losses.append(running_loss)
        print("Epoch:", epoch, "Loss:", running_loss)

    return model, losses

def train_conv_model(model, train_loader):

    import torch
    import torch.nn as nn
    import torch.optim as optim

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):

        running_loss = 0

        for images, _ in train_loader:

            outputs = model(images)

            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch:", epoch, "Loss:", running_loss)

    return model

def train_denoising(model, train_loader):

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.preprocess import add_noise

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):

        running_loss = 0

        for images, _ in train_loader:

            # add noise
            noisy = add_noise(images)

            # forward
            outputs = model(noisy)

            # compare with original
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch:", epoch, "Loss:", running_loss)

    return model