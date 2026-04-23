import inspect
import torch
import torch.nn as nn

class SimpleGeneModel(nn.Module):
    def __init__(self, promoter_len=400, promoter_channels=5, hidden_size=32, expr_dim=None):
        super().__init__()
        if expr_dim is None:
            raise ValueError("expr_dim must be provided, e.g. from dataset feature dimension")
        # promoter LSTM
        self.lstm = nn.LSTM(input_size=promoter_channels, hidden_size=hidden_size,
                            batch_first=True)
        # expression MLP
        self.expr_fc = nn.Linear(expr_dim, hidden_size)
        # output
        self.fc_out = nn.Linear(2 * hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, promoter, expr):
        # promoter: (batch, 400, 5)
        # expr: (batch, 16223)
        lstm_out, _ = self.lstm(promoter)        # (batch, 400, hidden)
        lstm_out = lstm_out[:, -1, :]            # take last time step -> (batch, hidden)
        expr_out = self.relu(self.expr_fc(expr)) # (batch, hidden)
        combined = torch.cat([lstm_out, expr_out], dim=1)  # (batch, 2*hidden)
        out = self.fc_out(combined)              # (batch, 1)
        return out


def _collect_model_registry():
    '''Collect all nn.Module subclasses defined in this module into a registry.'''
    registry = {}
    for name, obj in globals().items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
            if obj.__module__ == __name__:
                registry[name] = obj
    return registry


MODEL_REGISTRY = _collect_model_registry()


def get_model_class(model_name: str):
    if model_name not in MODEL_REGISTRY:
        choices = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model: {model_name}. Available models: {choices}")
    return MODEL_REGISTRY[model_name]


def build_model(model_name: str, **kwargs):
    model_cls = get_model_class(model_name)
    return model_cls(**kwargs)