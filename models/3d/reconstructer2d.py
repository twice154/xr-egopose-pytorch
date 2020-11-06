import torch
from torch import nn

class Reconstructer2D(nn.Module):
    def __init__(self, horizontal=48, vertical=48, channel=15, latent=20):
        super(Reconstructer2D, self).__init__()

        self.horizontal = horizontal
        self.vertical = vertical
        self.channel = channel
        self.latent = latent

        self.fc1 = nn.Linear(self.latent, 512)
        self.fc2 = nn.Linear(512, 2048)
        self.fc3 = nn.Linear(2048, (horizontal/8)*(vertical/8)*256)

        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 15, kernel_size=3, stride=2, padding=1)

    
    def forward(self, x):
