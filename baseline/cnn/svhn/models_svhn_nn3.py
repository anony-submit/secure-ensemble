import torch
import torch.nn as nn

class QuadraticActivation(nn.Module):
    def forward(self, x):
        return x ** 2

class SvhnNN3(nn.Module):
    def __init__(self):
        super(SvhnNN3, self).__init__()
        self.is_cnn_input = True
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2),
            QuadraticActivation(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            QuadraticActivation(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 64),
            QuadraticActivation(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
