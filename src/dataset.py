# purpose: define how to load one image+label pair

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image


class MelanomaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        transform=None
    ):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.transform = transform
    def __len__(self):
        return(len(self.df))
    
    def __getitem__(self,index):
        row=self.df.iloc[index]
        path=self.img_dir/ f"{row['image_name']}.jpg"
        img=Image.open(path)
        image=np.array(img.convert("RGB"))
        if self.transform is not None:
            image=self.transform(image=image)["image"]
        if "target" in self.df.columns:
            label = torch.tensor(row["target"], dtype=torch.float32)
        else:
            label = torch.tensor(-1.0, dtype=torch.float32)
        return image, label