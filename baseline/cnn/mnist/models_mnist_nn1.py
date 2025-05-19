import torch
import torch.nn as nn

class QuadraticActivation(nn.Module):
    def forward(self, x):
        return x ** 2

class MnistNN1(nn.Module):
    def __init__(self):
        super(MnistNN1, self).__init__()
        self.is_cnn_input = False
        self.fc1 = nn.Linear(784, 64)
        self.quad = QuadraticActivation()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.quad(self.fc1(x))
        return self.fc2(x)
