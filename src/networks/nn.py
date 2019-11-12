import torch.nn as nn
import torch.nn.functional as F


class BasicNN(nn.Module):
    """
    Implements a simple neural network architecture
    """
    def __init__(self, num_inputs, hidden_layer_size, num_outputs):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, num_outputs)

    def forward(self, input):
        h = F.relu(self.fc1(input.float()))
        return F.log_softmax(self.fc2(h))
