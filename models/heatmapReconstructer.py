import torch
from torch import nn

class HeatmapReconstructer(nn.Module):
    def __init__(self, horizontal=48, vertical=48, channel=15, latent=20):
        super(HeatmapReconstructer, self).__init__()

        self.horizontal = horizontal
        self.vertical = vertical
        self.channel = channel
        self.latent = latent

        self.fc1 = nn.Linear(self.latent, 512)
        self.lrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(512, 2048)
        self.fc3 = nn.Linear(2048, 6*6*256)

        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, bias=False, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, bias=False, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 15, kernel_size=4, bias=False, stride=2, padding=1)
        self.sig = nn.Sigmoid()

    
    def forward(self, x):  # (20)
        x = self.fc1(x)  # (512)
        x = self.lrelu(x)
        x = self.fc2(x)  # (2048)
        x = self.lrelu(x)
        x = self.fc3(x)  # (9216)
        x = self.lrelu(x)

        x = torch.reshape(x, (-1, 256, 6, 6))  # (256, 6, 6)
        x = self.conv1(x)  # (128, 12, 12)
        x = self.lrelu(x)
        x = self.conv2(x)  # (64, 24, 24)
        x = self.lrelu(x)
        x = self.conv3(x)  # (15, 48, 48)
        x = self.sig(x)

        return x