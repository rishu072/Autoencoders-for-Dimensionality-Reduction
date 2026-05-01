import torch
import torch.nn as nn

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # 🔹 Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)   # latent space
        )

        # 🔹 Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

import torch.nn as nn

class ConvAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # 🔹 Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28→14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 14→7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 7→4
            nn.ReLU()
        )

        # 🔹 Bottleneck
        self.fc1 = nn.Linear(128*4*4, 16)
        self.fc2 = nn.Linear(16, 128*4*4)

        # 🔹 Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # 4→7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 7→14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 14→28
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = x.view(-1, 128, 4, 4)

        x = self.decoder(x)

        return x