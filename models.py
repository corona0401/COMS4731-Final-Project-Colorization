import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class low_feature_net(nn.Module):
    def __init__(self):
        super(low_feature_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x


class global_feature_net(nn.Module):
    def __init__(self):
        super(global_feature_net, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 25088)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class mid_feature_net(nn.Module):
    def __init__(self):
        super(mid_feature_net, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class upsample_color_net(nn.Module):
    def __init__(self):
        super(upsample_color_net, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8 = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.upsample_nearest(x, scale_factor = 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.upsample_nearest(x, scale_factor = 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.upsample_nearest(x, scale_factor = 2)
        x = F.relu(self.conv7(x))
        x = F.tanh(self.conv8(x))
        return x
