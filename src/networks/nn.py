import torch.nn as nn
import torch.nn.functional as F


class BasicNN(nn.Module):
    """
    Implements a simple neural network architecture
    """
    def __init__(self, num_inputs, hidden_layer_sizes, num_outputs):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, hidden_layer_sizes[0])
        self.fc2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.fc3 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])
        self.fc4 = nn.Linear(hidden_layer_sizes[2], num_outputs)

    def forward(self, input):
        h = F.relu(self.fc1(input.float()))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        out = F.relu(self.fc4(h))
        return out
