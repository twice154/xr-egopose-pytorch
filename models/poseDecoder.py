import torch
from torch import nn

class PoseDecoder(nn.Module):
    def __init__(self, horizontal=48, vertical=48, channel=15, latent=20):
        super(PoseDecoder, self).__init__()

        self.horizontal = horizontal
        self.vertical = vertical
        self.channel = channel
        self.latent = latent

        self.fc1 = nn.Linear(self.latent, 32)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 48)
        self.tan = nn.Tanh()

    
    def forward(self, x):  # (20)
        x = self.fc1(x)  # (32)
        x = self.relu(x)
        x = self.fc2(x)  # (32)
        x = self.relu(x)
        x = self.fc3(x)  # (48)
        #################### Scaling이 -1~+1 사이로 되어있어서, Loss 계산시에 x2해서 해줘야함 (2x2x2 Cube 상에서 3DPose를 계산하기 떄문에)
        x = self.tan(x)  