import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from minist import get_mnist


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class NetV2(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Dropout(0.1),
        )
        self.block2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


# Default model alias used elsewhere
Net = NetV2
model = Net(num_classes=10)
num_parameters = sum(value.numel() for value in model.state_dict().values())
# print(f"{num_parameters = }")


def train(net, trainloader, optimizer, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    return net


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def run_centralised(epochs: int, lr: float, momentum: float = 0.9):
    """A minimal (but complete) training loop"""

    model = Net(num_classes=10)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    trainset, testset = get_mnist()
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=128)

    trained_model = train(model, trainloader, optim, epochs)
    loss, accuracy = test(trained_model, testloader)
    print(f"{loss = }")
    print(f"{accuracy = }")

# run_centralised(epochs=5, lr=0.01)
