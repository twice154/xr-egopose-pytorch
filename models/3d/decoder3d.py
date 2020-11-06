import torch
from torch import nn

class Decoder3D(nn.Module):
    def __init__(self, horizontal=48, vertical=48, channel=15, latent=20):
        super(Decoder3D, self).__init__()

        self.horizontal = horizontal
        self.vertical = vertical
        self.channel = channel
        self.latent = latent

        self.fc1 = nn.Linear(self.latent, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 48)

    
    def forward(self, x):
