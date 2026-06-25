import inspect
import torch
import torch.nn as nn
from typing import Any, Optional

from src.vae import SCVIEncoder

class SimpleGeneModel(nn.Module):
    def __init__(self, promoter_len: int = 400, promoter_channels: int = 5, hidden_size: int = 32, expr_dim: Optional[int] = None,
                 use_vae: bool = False, vae_encoder_path: Optional[str] = None, vae_fine_tune: bool = False, **kwargs: Any) -> None:
        super().__init__()
        if expr_dim is None:
            raise ValueError("expr_dim must be provided, e.g. from dataset feature dimension")
        # promoter LSTM
        self.lstm = nn.LSTM(input_size=promoter_channels, hidden_size=hidden_size,
                            batch_first=True)
        # expression MLP
        self.use_vae = use_vae
        self.vae_fine_tune = vae_fine_tune
        if use_vae and vae_encoder_path:
            self.vae_encoder = SCVIEncoder.from_pretrained(vae_encoder_path)
            if not vae_fine_tune:
                for p in self.vae_encoder.parameters():
                    p.requires_grad = False
            self.expr_fc = nn.Linear(self.vae_encoder.mean_encoder.out_features, hidden_size)
        else:
            self.expr_fc = nn.Linear(expr_dim, hidden_size)
        # output
        self.fc_out = nn.Linear(2 * hidden_size, 1)
        self.relu = nn.ReLU()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_vae and not self.vae_fine_tune:
            self.vae_encoder.eval()
        return self

    def forward(self, promoter: torch.Tensor, expr: torch.Tensor) -> torch.Tensor:
        # promoter: (batch, 400, 5)
        # expr: (batch, expr_dim)
        lstm_out, _ = self.lstm(promoter)        # (batch, 400, hidden)
        lstm_out = lstm_out[:, -1, :]            # take last time step -> (batch, hidden)
        self.last_lstm_out = lstm_out             # stored for monitoring
        if self.use_vae:
            expr = self.vae_encoder(expr)         # (batch, n_latent)
        expr_out = self.relu(self.expr_fc(expr)) # (batch, hidden)
        self.last_expr_out = expr_out             # stored for monitoring
        combined = torch.cat([lstm_out, expr_out], dim=1)  # (batch, 2*hidden)
        out = self.fc_out(combined)              # (batch, 1)
        return out


class LSTMmodel(nn.Module):
    '''More complex model with multi-layer bidirectional LSTM, deeper MLP, and dropout.

    fusion: "concat" (default) concatenates LSTM and expression branches.
            "gate" uses a promoter-driven gate to control expression features element-wise
            before concatenation: context_gate = sigmoid(Linear(lstm_out)), fused = expr_out * context_gate.'''

    def __init__(self, promoter_len: int = 400, promoter_channels: int = 5, hidden_size: int = 64, expr_dim: Optional[int] = None,
                 dropout: float = 0.1, lstm_layers: int = 2,
                 use_vae: bool = False, vae_encoder_path: Optional[str] = None, vae_fine_tune: bool = False,
                 output_mode: str = "scalar", fusion: str = "concat", **kwargs: Any) -> None:
        super().__init__()
        if expr_dim is None:
            raise ValueError("expr_dim must be provided, e.g. from dataset feature dimension")

        self.lstm = nn.LSTM(
            input_size=promoter_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        lstm_out_dim = hidden_size * 2  # bidirectional

        self.attn = nn.Linear(lstm_out_dim, 1)  # attention over LSTM time steps
        # add full connection layer, GELU activation function layer, normalization and dropout after attention pooling
        self.lstm_fc = nn.Linear(lstm_out_dim, hidden_size * 2)
        self.gelu = nn.GELU()
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)
        self.lstm_dropout = nn.Dropout(dropout)

        self.use_vae = use_vae
        self.vae_fine_tune = vae_fine_tune
        if use_vae and vae_encoder_path:
            self.vae_encoder = SCVIEncoder.from_pretrained(vae_encoder_path)
            if not vae_fine_tune:
                for p in self.vae_encoder.parameters():
                    p.requires_grad = False
            self.expr_fc = nn.Linear(self.vae_encoder.mean_encoder.out_features, hidden_size * 2)
        else:
            self.expr_fc = nn.Linear(expr_dim, hidden_size * 2)
        self.expr_norm = nn.LayerNorm(hidden_size * 2)  # add normalization layer for VAE branch
        self.expr_dropout = nn.Dropout(dropout)

        self.fusion = fusion
        if fusion == "gate":
            self.seq_gate = nn.Linear(lstm_out_dim, hidden_size * 2)

        self.fc_norm = nn.LayerNorm(2 * hidden_size)  # normalization layer for combined features
        self.fc1 = nn.Linear(lstm_out_dim + hidden_size * 2, hidden_size * 2)
        self.fc1_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2_dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.output_mode = output_mode
        if output_mode == "zinb":
            self.mu_head = nn.Linear(hidden_size, 1)
            self.theta_head = nn.Linear(hidden_size, 1)
            self.pi_head = nn.Linear(hidden_size, 1)
        else:
            self.fc_out = nn.Linear(hidden_size, 1)


    def forward(self, promoter: torch.Tensor, expr: torch.Tensor):
        # promoter: (batch, 400, 5)
        # expr: (batch, expr_dim)
        lstm_out, _ = self.lstm(promoter)           # (batch, 400, 2*hidden)
        # Attention pooling: sum(alpha_t * h_t)
        attn_scores = self.attn(lstm_out)           # (batch, 400, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, 400, 1)
        lstm_out = (attn_weights * lstm_out).sum(dim=1)   # (batch, 2*hidden)
        lstm_out = self.lstm_norm(self.lstm_fc(lstm_out))  # (batch, 2*hidden)
        lstm_out = self.gelu(lstm_out)
        lstm_out = self.lstm_dropout(lstm_out)
        self.last_lstm_out = lstm_out                     # stored for monitoring

        if self.use_vae:
            expr = self.vae_encoder(expr)             # (batch, n_latent)
        expr_out = self.expr_norm(self.expr_fc(expr))     # (batch, 2*hidden)
        expr_out = self.gelu(expr_out)
        expr_out = self.expr_dropout(expr_out)
        self.last_expr_out = expr_out                     # stored for monitoring

        if self.fusion == "gate":
            context_gate = torch.sigmoid(self.seq_gate(lstm_out))  # (batch, 2*hidden)
            fused_expr = expr_out * context_gate                    # element-wise gate
            combined = torch.cat([lstm_out, fused_expr], dim=1)    # (batch, 4*hidden)
        else:
            combined = torch.cat([lstm_out, expr_out], dim=1)       # (batch, 4*hidden)
        x = self.fc_norm(self.fc1(combined))          # (batch, 2*hidden)
        x = self.relu(x)
        x = self.fc1_dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc2_dropout(x)

        if self.output_mode == "zinb":
            mu_ratio = torch.exp(torch.clamp(self.mu_head(x), max=10))     # (batch, 1)
            theta = torch.exp(torch.clamp(self.theta_head(x), max=10))     # (batch, 1)
            pi = torch.sigmoid(self.pi_head(x))                             # (batch, 1)
            return mu_ratio, theta, pi
        else:
            return self.fc_out(x)                       # (batch, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_vae and not self.vae_fine_tune:
            self.vae_encoder.eval()
        return self


class ConvAttentionModel(nn.Module):
    """Promoter CNN + Attention pooling + Expression MLP to fusion."""

    def __init__(self, promoter_len: int = 400, promoter_channels: int = 5, hidden_size: int = 128, expr_dim: Optional[int] = None, dropout: float = 0.3,
                 use_vae: bool = False, vae_encoder_path: Optional[str] = None, vae_fine_tune: bool = False,
                 output_mode: str = "scalar", **kwargs: Any) -> None:
        super().__init__()
        if expr_dim is None:
            raise ValueError("expr_dim must be provided")

        # Promoter branch: Conv1D → ReLU → Conv1D → ReLU → Attention pooling → Dense(128)
        self.conv1 = nn.Conv1d(promoter_channels, 64, kernel_size=8, padding="same")
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, padding="same")
        self.promoter_attn = nn.Linear(128, 1)  # attention over conv positions
        self.promoter_fc = nn.Linear(128, 128)

        # Expression branch: (VAE?) → Dense(512) → ReLU → Dense(256)
        self.use_vae = use_vae
        self.vae_fine_tune = vae_fine_tune
        if use_vae and vae_encoder_path:
            self.vae_encoder = SCVIEncoder.from_pretrained(vae_encoder_path)
            if not vae_fine_tune:
                for p in self.vae_encoder.parameters():
                    p.requires_grad = False
            self.expr_fc1 = nn.Linear(self.vae_encoder.mean_encoder.out_features, 512)
        else:
            self.expr_fc1 = nn.Linear(expr_dim, 512)
        self.expr_dropout1 = nn.Dropout(dropout)
        self.expr_fc2 = nn.Linear(512, 256)
        self.expr_dropout2 = nn.Dropout(dropout)

        # Fusion: concat(128 + 256) → Dense(128) → ReLU → (scalar or ZINB)
        self.fc1 = nn.Linear(128 + 256, 128)
        self.fc_norm = nn.LayerNorm(128)
        self.fc1_dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc2_dropout = nn.Dropout(dropout)

        self.output_mode = output_mode
        if output_mode == "zinb":
            self.mu_head = nn.Linear(128, 1)
            self.theta_head = nn.Linear(128, 1)
            self.pi_head = nn.Linear(128, 1)
        else:
            self.fc_out = nn.Linear(128, 1)

    def forward(self, promoter: torch.Tensor, expr: torch.Tensor):
        # Promoter branch
        p = promoter.permute(0, 2, 1)              # (batch, 5, 400)
        p = self.relu(self.conv1(p))               # (batch, 64, 400)
        p = self.relu(self.conv2(p))               # (batch, 128, 400)
        p = p.permute(0, 2, 1)                     # (batch, 400, 128)
        # Attention pooling over sequence positions
        attn_scores = self.promoter_attn(p)         # (batch, 400, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        p = (attn_weights * p).sum(dim=1)           # (batch, 128)
        p = self.relu(self.promoter_fc(p))           # (batch, 128)

        # Expression branch
        if self.use_vae:
            expr = self.vae_encoder(expr)             # (batch, n_latent)
        e = self.relu(self.expr_fc1(expr))           # (batch, 512)
        e = self.expr_dropout1(e)
        e = self.relu(self.expr_fc2(e))              # (batch, 256)
        e = self.expr_dropout2(e)

        # Fusion
        combined = torch.cat([p, e], dim=1)          # (batch, 128+256=384)
        x = self.fc_norm(self.fc1(combined))          # (batch, 2*hidden)
        x = self.relu(x)
        x = self.fc1_dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc2_dropout(x)

        if self.output_mode == "zinb":
            mu_ratio = torch.exp(torch.clamp(self.mu_head(x), max=10))     # (batch, 1)
            theta = torch.exp(torch.clamp(self.theta_head(x), max=10))     # (batch, 1)
            pi = torch.sigmoid(self.pi_head(x))                             # (batch, 1)
            return mu_ratio, theta, pi
        else:
            return self.fc_out(x)                       # (batch, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_vae and not self.vae_fine_tune:
            self.vae_encoder.eval()
        return self


class CNNFlattenPromoterModel(nn.Module):
    """Promoter CNN-flatten encoder fused with a masked expression MLP."""

    def __init__(
        self,
        promoter_len: int = 400,
        promoter_channels: int = 5,
        hidden_size: int = 128,
        expr_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_vae: bool = False,
        vae_encoder_path: Optional[str] = None,
        vae_fine_tune: bool = False,
        output_mode: str = "scalar",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if expr_dim is None:
            raise ValueError("expr_dim must be provided")

        conv_width = max(16, min(64, hidden_size))
        self.promoter_conv = nn.Sequential(
            nn.Conv1d(promoter_channels, conv_width, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_width, conv_width, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.promoter_fc = nn.Sequential(
            nn.Linear(promoter_len * conv_width, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.use_vae = use_vae
        self.vae_fine_tune = vae_fine_tune
        if use_vae and vae_encoder_path:
            self.vae_encoder = SCVIEncoder.from_pretrained(vae_encoder_path)
            if not vae_fine_tune:
                for p in self.vae_encoder.parameters():
                    p.requires_grad = False
            expr_in = self.vae_encoder.mean_encoder.out_features
        else:
            expr_in = expr_dim

        self.expr_mlp = nn.Sequential(
            nn.Linear(expr_in, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_mode = output_mode
        if output_mode == "zinb":
            self.mu_head = nn.Linear(hidden_size, 1)
            self.theta_head = nn.Linear(hidden_size, 1)
            self.pi_head = nn.Linear(hidden_size, 1)
        else:
            self.fc_out = nn.Linear(hidden_size, 1)

    def encode_promoter(self, promoter: torch.Tensor) -> torch.Tensor:
        x = promoter.permute(0, 2, 1)
        x = self.promoter_conv(x)
        x = x.flatten(start_dim=1)
        return self.promoter_fc(x)

    def forward(self, promoter: torch.Tensor, expr: torch.Tensor):
        promoter_out = self.encode_promoter(promoter)
        self.last_lstm_out = promoter_out
        if self.use_vae:
            expr = self.vae_encoder(expr)
        expr_out = self.expr_mlp(expr)
        self.last_expr_out = expr_out
        x = self.fusion(torch.cat([promoter_out, expr_out], dim=1))

        if self.output_mode == "zinb":
            mu_ratio = torch.exp(torch.clamp(self.mu_head(x), max=10))
            theta = torch.exp(torch.clamp(self.theta_head(x), max=10))
            pi = torch.sigmoid(self.pi_head(x))
            return mu_ratio, theta, pi
        return self.fc_out(x)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_vae and not self.vae_fine_tune:
            self.vae_encoder.eval()
        return self


class PromoterBaseline(nn.Module):
    """Promoter-only baseline: CNN → attention pooling → output."""

    def __init__(self, promoter_len: int = 400, promoter_channels: int = 5, hidden_size: int = 128, expr_dim: Optional[int] = None, dropout: float = 0.3, **kwargs: Any) -> None:
        super().__init__()
        # Same CNN backbone as ConvAttentionModel promoter branch
        self.conv1 = nn.Conv1d(promoter_channels, 64, kernel_size=8, padding="same")
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, padding="same")
        self.attn = nn.Linear(128, 1)
        self.fc1 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, promoter: torch.Tensor, expr: torch.Tensor) -> torch.Tensor:
        x = promoter.permute(0, 2, 1)              # (batch, 5, 400)
        x = self.relu(self.conv1(x))               # (batch, 64, 400)
        x = self.relu(self.conv2(x))               # (batch, 128, 400)
        x = x.permute(0, 2, 1)                     # (batch, 400, 128)
        # Attention pooling
        attn_scores = self.attn(x)                  # (batch, 400, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        x = (attn_weights * x).sum(dim=1)           # (batch, 128)
        x = self.relu(self.fc1(x))
        out = self.fc_out(x)                        # (batch, 1)
        return out


class ExpressionBaseline(nn.Module):
    """Expression-only baseline: MLP to output."""

    def __init__(self, promoter_len: int = 400, promoter_channels: int = 5, hidden_size: int = 128, expr_dim: Optional[int] = None, dropout: float = 0.3,
                 use_vae: bool = False, vae_encoder_path: Optional[str] = None, vae_fine_tune: bool = False, **kwargs: Any) -> None:
        super().__init__()
        if expr_dim is None:
            raise ValueError("expr_dim must be provided")
        # Expression branch: (VAE?) → Dense(512) → ReLU → Dense(256) → output
        self.use_vae = use_vae
        if use_vae and vae_encoder_path:
            self.vae_encoder = SCVIEncoder.from_pretrained(vae_encoder_path)
            if not vae_fine_tune:
                for p in self.vae_encoder.parameters():
                    p.requires_grad = False
            self.fc1 = nn.Linear(self.vae_encoder.mean_encoder.out_features, 512)
        else:
            self.fc1 = nn.Linear(expr_dim, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, promoter: torch.Tensor, expr: torch.Tensor) -> torch.Tensor:
        if self.use_vae:
            expr = self.vae_encoder(expr)             # (batch, n_latent)
        x = self.relu(self.fc1(expr))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        out = self.fc_out(x)
        return out


## Model registry utilities
def _collect_model_registry() -> dict:
    '''Collect all nn.Module subclasses defined in this module into a registry.'''
    registry = {}
    for name, obj in globals().items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
            if obj.__module__ == __name__:
                registry[name] = obj
    return registry


MODEL_REGISTRY = _collect_model_registry()


def get_model_class(model_name: str) -> type:
    if model_name not in MODEL_REGISTRY:
        choices = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model: {model_name}. Available models: {choices}")
    return MODEL_REGISTRY[model_name]


def build_model(model_name: str, **kwargs: Any) -> nn.Module:
    model_cls = get_model_class(model_name)
    return model_cls(**kwargs)
