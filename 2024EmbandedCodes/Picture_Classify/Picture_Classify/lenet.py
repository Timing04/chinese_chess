import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6,
                            kernel_size=5, stride=1, padding=2)
        self.r1 = nn.ReLU(inplace=True)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, 5, 1)
        self.r3 = nn.ReLU(inplace=True)
        self.s4 = nn.MaxPool2d(2, 2)

        # 遇到了卷积层变为全连接层
        self.c5 = nn.Linear(16*5*5, 120)
        self.r5 = nn.ReLU(inplace=True)
        self.f6 = nn.Linear(120, 84)
        self.r6 = nn.ReLU(inplace=True)
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):     # 输入 1*28*28
        out = self.c1(x)      # 6*28*28
        out = self.r1(out)    # 6*28*28
        out = self.s2(out)    # 6*14*14
        out = self.c3(out)    # 16*10*10
        out = self.r3(out)    # 16*10*10
        out = self.s4(out)    # 16*5*5

        out = out.view(-1, 16*5*5)  # out.size()[0], 1*400
        out = self.c5(out)    # 400 --> 120
        out = self.r5(out)
        out = self.f6(out)    # 120 --> 84
        out = self.r6(out)
        out = self.f7(out)    # 84 --> 10
        return out


if __name__ == "__main__":
    net = LeNet()
    print(net)