import torch
import torch.nn as nn

class QuadraticActivation(nn.Module):
    def forward(self, x):
        return x * x

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
       
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, stride=2)
        self.quad1 = QuadraticActivation()
        self.fc1 = nn.Linear(784, 64)  # 14x14x4 = 784
        self.quad2 = QuadraticActivation()
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.conv(x)  # -> batch_size x 4 x 14 x 14
        x = x.view(x.size(0), -1)  # -> batch_size x 784
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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.layer1 = self.make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*BasicBlock.expansion, 10)

    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out