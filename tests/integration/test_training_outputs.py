from __future__ import annotations

import pytest
from torch.utils.data import DataLoader

from scripts.train import train_model
from src.dataset import MyDataset
from src.model import build_model
import src.utils as utils


pytestmark = pytest.mark.integration


def test_tiny_training_writes_expected_outputs(tiny_data_dir, tmp_path, monkeypatch) -> None:
    train_dataset = MyDataset(tiny_data_dir / "promoter_train.csv", tiny_data_dir / "integrated_data.h5ad")
    val_dataset = MyDataset(tiny_data_dir / "promoter_val.csv", tiny_data_dir / "integrated_data.h5ad")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    model = build_model("SimpleGeneModel", expr_dim=train_dataset.X.shape[1], hidden_size=2)

    def prepare_output_dirs(_base_dir, exp_name):
        run_dir = tmp_path / exp_name
        ckpt_dir = run_dir / "checkpoints"
        plots_dir = run_dir / "plots"
        log_dir = run_dir / "log"
        for path in [ckpt_dir, plots_dir, log_dir]:
            path.mkdir(parents=True, exist_ok=True)
        return run_dir, ckpt_dir, plots_dir, log_dir

    monkeypatch.setattr(utils, "_prepare_output_dirs", prepare_output_dirs)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        exp_name="tiny_train",
        epochs=6,
        learning_rate=1e-4,
        patience=0,
        loss_type="mse",
    )

    run_dir = tmp_path / "tiny_train"
    assert (run_dir / "checkpoints" / "last.ckpt").exists()
    assert (run_dir / "checkpoints" / "best_model.safetensors").exists()
    assert (run_dir / "log" / "train_log.csv").exists()
    assert (run_dir / "log" / "step_train_loss.csv").exists()


def test_tiny_zinb_training_amp_flag_smoke(tiny_data_dir, tmp_path, monkeypatch) -> None:
    train_dataset = MyDataset(tiny_data_dir / "promoter_train.csv", tiny_data_dir / "integrated_data.h5ad")
    val_dataset = MyDataset(tiny_data_dir / "promoter_val.csv", tiny_data_dir / "integrated_data.h5ad")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    model = build_model("LSTMmodel", expr_dim=train_dataset.X.shape[1], hidden_size=4, output_mode="zinb")

    def prepare_output_dirs(_base_dir, exp_name):
        run_dir = tmp_path / exp_name
        ckpt_dir = run_dir / "checkpoints"
        plots_dir = run_dir / "plots"
        log_dir = run_dir / "log"
        for path in [ckpt_dir, plots_dir, log_dir]:
            path.mkdir(parents=True, exist_ok=True)
        return run_dir, ckpt_dir, plots_dir, log_dir

    monkeypatch.setattr(utils, "_prepare_output_dirs", prepare_output_dirs)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        exp_name="tiny_zinb_amp",
        epochs=6,
        learning_rate=1e-4,
        patience=0,
        loss_type="zinb",
        amp=True,
    )

    run_dir = tmp_path / "tiny_zinb_amp"
    log_file = run_dir / "log" / "train_log.csv"
    step_log_file = run_dir / "log" / "step_train_loss.csv"
    assert log_file.exists()
    assert step_log_file.exists()
    assert "nan" not in step_log_file.read_text(encoding="utf-8").lower()
