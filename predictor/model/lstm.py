import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=1, use_bidirectional=False, use_dropout=False):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=use_bidirectional,
                           dropout=0.5 if use_dropout else 0.)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]),
                                        dim=1))

        return self.fc(hidden.squeeze(0))

