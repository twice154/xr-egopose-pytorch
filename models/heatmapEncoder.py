import torch
from torch import nn

class HeatmapEncoder(nn.Module):
    def __init__(self, horizontal=48, vertical=48, channel=15, latent=20):
        super(HeatmapEncoder, self).__init__()

        self.horizontal = horizontal
        self.vertical = vertical
        self.channel = channel
        self.latent = latent

        self.conv1 = nn.Conv2d(self.channel, 64, kernel_size=4, bias=False, stride=2, padding=1)
        self.convbn1 = nn.BatchNorm2d(64)
        self.lrelu = nn.LeakyReLU(0.2)
        # self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, bias=False, stride=2, padding=1)
        self.convbn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, bias=False, stride=2, padding=1)
        self.convbn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(6*6*256, 2048)
        self.fcbn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fcbn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, self.latent)

    
    def forward(self, x):  # (15, 48, 48)
        x = self.conv1(x)  # (64, 24, 24)
        x = self.convbn1(x)
        x = self.lrelu(x)
        # x = self.dropout(x)
        x = self.conv2(x)  # (128, 12, 12)
        x = self.convbn2(x)
        x = self.lrelu(x)
        # x = self.dropout(x)
        x = self.conv3(x)  # (256, 6, 6)
        x = self.convbn3(x)
        x = self.lrelu(x)
        # x = self.dropout(x)

        x = torch.flatten(x, 1)  # (256 x 6 x 6) = (18432 / 2) = (9216)
        x = self.fc1(x)  # (2048)
        x = self.fcbn1(x)
        x = self.lrelu(x)
        # x = self.dropout(x)
        x = self.fc2(x)  # (512)
        x = self.fcbn2(x)
        x = self.lrelu(x)
        # x = self.dropout(x)
        x = self.fc3(x)  # (20)

        return x