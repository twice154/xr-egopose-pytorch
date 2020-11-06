import torch
from torch import nn

class Encoder3D(nn.Module):
    def __init__(self, horizontal=48, vertical=48, channel=15, latent=20):
        super(Encoder3D, self).__init__()

        self.horizontal = horizontal
        self.vertical = vertical
        self.channel = channel
        self.latent = latent

        self.conv1 = nn.Conv2d(self.channel, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear((horizontal/8)*(vertical/8)*256, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, self.latent)

    
    def forward(self, x):
