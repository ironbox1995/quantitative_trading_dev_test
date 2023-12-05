import torch.nn as nn


# Define a regression neural network
class FullyConnectedRegressionNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_prob):
        super(FullyConnectedRegressionNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_prob))
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # Output layer for regression (output dimension = 1)
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_prob))
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x