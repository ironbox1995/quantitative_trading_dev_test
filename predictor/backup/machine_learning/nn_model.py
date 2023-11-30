import torch.nn as nn

from machine_learning.model_config import *


# Define the fully connected neural network with dropout
class FCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, dropout_rate=DROPOUT):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(num_hidden_layers)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout_rate) for i in range(num_hidden_layers)])
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout1(out)
        out = self.relu(out)
        for layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            out = layer(out)
            out = dropout_layer(out)
            out = self.relu(out)
        out = self.fc2(out)
        out = out.squeeze(1)
        return out
