import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset import MelanomaDataset
from src.transforms import get_transforms


def create_fake_dataset(num_images=10):
    """
    Create temporary images + dataframe for testing.
    """
    tmp_dir = tempfile.TemporaryDirectory()
    img_dir = Path(tmp_dir.name)

    image_names = []

    for i in range(num_images):
        name = f"FAKE_{i:04d}"
        image_names.append(name)

        img = Image.fromarray(
            np.random.randint(
                0,
                255,
                (224, 224, 3),
                dtype=np.uint8
            )
        )

        img.save(img_dir / f"{name}.jpg")

    df = pd.DataFrame({
        "image_name": image_names,
        "target": [i % 2 for i in range(num_images)]
    })

    return tmp_dir, img_dir, df


def test_dataset_length():

    tmp_dir, img_dir, df = create_fake_dataset()

    dataset = MelanomaDataset(
        df=df,
        img_dir=img_dir,
        transform=None
    )

    assert len(dataset) == len(df)

    tmp_dir.cleanup()


def test_dataset_without_transforms():

    tmp_dir, img_dir, df = create_fake_dataset()

    dataset = MelanomaDataset(
        df=df,
        img_dir=img_dir,
        transform=None
    )

    image, label = dataset[0]

    assert isinstance(image, np.ndarray)
    assert image.shape == (224, 224, 3)

    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.float32

    tmp_dir.cleanup()


def test_dataset_with_transforms():

    tmp_dir, img_dir, df = create_fake_dataset()

    transform = get_transforms(
        phase="val",
        size=224
    )

    dataset = MelanomaDataset(
        df=df,
        img_dir=img_dir,
        transform=transform
    )

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)

    assert image.shape == (3, 224, 224)

    assert image.dtype == torch.float32

    assert label.dtype == torch.float32

    tmp_dir.cleanup()


def test_dataloader_batching():

    tmp_dir, img_dir, df = create_fake_dataset()

    transform = get_transforms(
        phase="val",
        size=224
    )

    dataset = MelanomaDataset(
        df=df,
        img_dir=img_dir,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    images, labels = next(iter(loader))

    assert images.shape == (4, 3, 224, 224)

    assert labels.shape == (4,)

    tmp_dir.cleanup()


def test_dataset_without_target_column():

    tmp_dir, img_dir, df = create_fake_dataset()

    df = df.drop(columns=["target"])

    dataset = MelanomaDataset(
        df=df,
        img_dir=img_dir,
        transform=None
    )

    _, label = dataset[0]

    assert label.item() == -1.0

    tmp_dir.cleanup()