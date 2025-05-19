import torch
import torch.nn as nn

class QuadraticActivation(nn.Module):
    def forward(self, x):
        return x ** 2

class FmnistNN2(nn.Module):
    def __init__(self):
        super(FmnistNN2, self).__init__()
        self.is_cnn_input = True
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, stride=2)
        self.quad1 = QuadraticActivation()
        self.fc1 = nn.Linear(784, 64)  # 14x14x4 = 784
        self.quad2 = QuadraticActivation()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv(x)                  # (batch_size, 4, 14, 14)
        x = x.view(x.size(0), -1)         # (batch_size, 784)
        x = self.quad1(x)
        x = self.fc1(x)
        x = self.quad2(x)
        x = self.fc2(x)
        return x

    def get_all_params(self):
        kernels = []
        for i in range(4):
            kernel = self.conv.weight[i, 0].cpu().detach().numpy().flatten().tolist()
            kernels.append(kernel)

        return {
            "conv_params": {
                "kernels": kernels,
                "bias": self.conv.bias.cpu().detach().numpy().tolist()
            },
            "fc1.weight": self.fc1.weight.cpu().detach().numpy().tolist(),
            "fc1.bias": self.fc1.bias.cpu().detach().numpy().tolist(),
            "fc2.weight": self.fc2.weight.cpu().detach().numpy().tolist(),
            "fc2.bias": self.fc2.bias.cpu().detach().numpy().tolist()
        }
