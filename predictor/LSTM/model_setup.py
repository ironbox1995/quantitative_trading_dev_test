import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        hidden_cell = (torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(input_seq.device),
                       torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(input_seq.device))

        lstm_out, _ = self.lstm(input_seq, hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

