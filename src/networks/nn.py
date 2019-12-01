import torch.nn as nn
import torch.nn.functional as F


class BasicNN(nn.Module):
    """
    Implements a simple neural network architecture
    """
    def __init__(self, num_inputs, hidden_layer_size, num_outputs, task):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, num_outputs)
        self.task = task

    def forward(self, input):
        h = F.relu(self.fc1(input.float()))
        fc2_output = self.fc2(h)

        if self.task.lower() == 'classification':
            return F.log_softmax(fc2_output)
        # regression
        return fc2_output
