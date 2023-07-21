import torch.nn as nn


# Step 3: Implement the "Net" architecture
class DilatedDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DilatedDepthwiseSeparableConv, self).__init__()
        self.dilated_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.dilated_conv(x)
        x = self.pointwise_conv(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            DilatedDepthwiseSeparableConv(64, 128, kernel_size=3, padding=2, dilation=2),
            DilatedDepthwiseSeparableConv(128, 256, kernel_size=3, padding=4, dilation=4),
            DilatedDepthwiseSeparableConv(256, 40, kernel_size=3, padding=6, dilation=6)
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(40, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = Net()
