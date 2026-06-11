import random
from anyio import Path
import numpy as np
import torch
import os

def seed_everything(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_checkpoint(model, fold: int, path, epoch: int, auc: float):
    torch.save({
        "fold":        fold,
        "epoch":       epoch,
        "auc":         auc,
        "model_state": model.state_dict(),
    }, path)
    print(f"  → Checkpoint saved: {path.name} (AUC={auc:.4f})")

def load_checkpoint(model, path: Path, device):
    """Load weights back into a model for inference."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from fold {checkpoint['fold']}, "
          f"epoch {checkpoint['epoch']}, AUC={checkpoint['auc']:.4f}")
    return model