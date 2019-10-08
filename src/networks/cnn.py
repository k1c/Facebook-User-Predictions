import torch.nn as nn
import torch.nn.functional as F


class BasicNet(nn.Module):
    """
    Implements a simple CNN model architecture for gender prediction.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 32 * 32, 2)

    def forward(self, input):
        # First convolutional block
        h = F.relu(self.conv1(input))
        h = F.relu(self.conv2(h))

        # Pooling layer
        h = F.max_pool2d(h, 2)

        # Second convolutional block
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        # Pooling layer
        h = F.max_pool2d(h, 2)

        # Fully connected layer
        h = h.view(-1, 64 * 32 * 32)
        out = F.relu(self.fc(h))

        return out
