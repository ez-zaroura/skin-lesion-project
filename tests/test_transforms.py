import numpy as np
import torch

from src.transforms import (
    get_transforms,
    get_tta_transforms
)


def test_train_transform_shape():

    image = np.random.randint(
        0, 255,
        (512, 512, 3),
        dtype=np.uint8
    )

    transform = get_transforms(
        "train",
        size=380
    )

    output = transform(
        image=image
    )["image"]

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 380, 380)


def test_val_transform_shape():

    image = np.random.randint(
        0, 255,
        (512, 512, 3),
        dtype=np.uint8
    )

    transform = get_transforms(
        "val",
        size=380
    )

    output = transform(
        image=image
    )["image"]

    assert output.shape == (3, 380, 380)


def test_tta_transform():

    image = np.random.randint(
        0, 255,
        (512, 512, 3),
        dtype=np.uint8
    )

    tta = get_tta_transforms(
        size=380
    )

    outputs = [
        tta(image=image)["image"]
        for _ in range(5)
    ]

    different = any(
        not torch.allclose(outputs[0], x)
        for x in outputs[1:]
    )

    assert different