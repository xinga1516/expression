import json
from pathlib import Path

import torch
import torch.nn as nn


class SCVIEncoder(nn.Module):
    """Replicates scvi.nn.Encoder architecture for weight loading.

    Matches: FCLayers(n_input -> n_hidden, n_layers) -> Linear(n_hidden -> n_latent).
    Designed so that encoder.pt saved from a trained scVI model can be loaded
    with load_state_dict(..., strict=False).
    """

    def __init__(self, n_input, n_latent=10, n_hidden=128, n_layers=1, dropout_rate=0.1):
        super().__init__()
        blocks = []
        in_dim = n_input
        for _ in range(n_layers):
            blocks.append(nn.Sequential(
                nn.Linear(in_dim, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
            ))
            in_dim = n_hidden
        self.encoder = nn.ModuleList(blocks)
        self.mean_encoder = nn.Linear(n_hidden, n_latent)

    def forward(self, x):
        for block in self.encoder:
            x = block(x)
        return self.mean_encoder(x)

    @classmethod
    def from_pretrained(cls: type['SCVIEncoder'], checkpoint_dir: Path, device="cpu"):
        """Load a pretrained SCVIEncoder from an scVI output directory.

        Expects: {checkpoint_dir}/config.json and {checkpoint_dir}/encoder.pt
        """
        ckpt_dir = Path(checkpoint_dir)
        config_path = ckpt_dir / "config.json"
        encoder_path = ckpt_dir / "encoder.pt"

        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")
        if not encoder_path.exists():
            raise FileNotFoundError(f"encoder.pt not found in {checkpoint_dir}")

        with open(config_path) as f:
            cfg = json.load(f)

        model = cls(
            n_input=cfg["n_input"],
            n_latent=cfg["n_latent"],
            n_hidden=cfg.get("n_hidden", 128),
            n_layers=cfg.get("n_layers", 1),
            dropout_rate=cfg.get("dropout_rate", 0.1),
        )

        encoder_state = torch.load(encoder_path, map_location=device, weights_only=True)
        # Remap scVI key names to our SCVIEncoder key names:
        #   scVI:  encoder.fc_layers.Layer 0.0.weight
        #   ours:  encoder.0.0.weight
        # Also skip var_encoder keys (only used during VAE training, not for downstream)
        remapped = {}
        for k, v in encoder_state.items():
            if k.startswith("var_encoder"):
                continue
            if "encoder.fc_layers.Layer " in k:
                k = k.replace("encoder.fc_layers.Layer ", "encoder.")
            remapped[k] = v
        model.load_state_dict(remapped, strict=False)
        model.to(device)
        model.eval()
        return model
