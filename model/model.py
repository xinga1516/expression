import torch
import torch.nn as nn

class SimpleGeneModel(nn.Module):
    def __init__(self, promoter_len=400, promoter_channels=5, 
                 lstm_hidden=32, expr_dim=16262):
        super().__init__()
        # promoter LSTM
        self.lstm = nn.LSTM(input_size=promoter_channels, hidden_size=lstm_hidden,
                            batch_first=True)
        # expression MLP
        self.expr_fc = nn.Linear(expr_dim, lstm_hidden)
        # output
        self.fc_out = nn.Linear(2 * lstm_hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, promoter, expr):
        # promoter: (batch, 400, 5)
        # expr: (batch, 16300)
        lstm_out, _ = self.lstm(promoter)        # (batch, 400, hidden)
        lstm_out = lstm_out[:, -1, :]            # take last time step -> (batch, hidden)
        expr_out = self.relu(self.expr_fc(expr)) # (batch, hidden)
        combined = torch.cat([lstm_out, expr_out], dim=1)  # (batch, 2*hidden)
        out = self.fc_out(combined)              # (batch, 1)
        return out