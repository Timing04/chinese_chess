import torch.nn as nn

class LeCNN(nn.Module):
    def __init__(self):
        super(LeCNN, self).__init__()
        self.net = nn.Sequential(
            # 3*224*224
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 6 224 224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # 6 112 112
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1
            ),
            # 16 108 108
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # 16 54 54
            nn.Flatten(),
            nn.Linear(
                in_features=16*54*54,
                out_features=1200
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=1200,
                out_features=84
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=84,
                out_features=2
            ),

        )

    def forward(self, x):
        y = self.net(x)
        return y
