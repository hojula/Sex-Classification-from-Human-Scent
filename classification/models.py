
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=1, *args, **kwargs):
        super(LeNet5, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(962192, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out[:, 0]
        return out

class LinearProjectionModel1(nn.Module):  # err 13.5
    def __init__(self, output_size=1, *args, **kwargs):
        super(LinearProjectionModel1, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
        )
        self.flatten = nn.Flatten()
        print("Output size: ", output_size)
        self.fc = nn.Linear(15189, output_size)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x[:, 0]