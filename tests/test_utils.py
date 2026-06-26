from pathlib import Path

import torch

from src.utils import (
    seed_everything,
    get_device,
    save_checkpoint,
    load_checkpoint
)


def test_seed_everything():
    seed_everything(42)

    a = torch.rand(1)

    seed_everything(42)

    b = torch.rand(1)

    assert torch.equal(a, b)


def test_get_device():
    device = get_device()

    assert str(device) in ["cpu", "cuda"]


def test_save_and_load_checkpoint():

    model = torch.nn.Linear(4, 1)

    checkpoint_path = Path(
        "outputs/test_checkpoint.pth"
    )

    save_checkpoint(
        model=model,
        fold=1,
        path=checkpoint_path,
        epoch=3,
        auc=0.91
    )

    assert checkpoint_path.exists()

    loaded_model = torch.nn.Linear(4, 1)

    device = get_device()

    load_checkpoint(
        loaded_model,
        checkpoint_path,
        device
    )

    assert torch.allclose(
        model.weight,
        loaded_model.weight
    )

    checkpoint_path.unlink()