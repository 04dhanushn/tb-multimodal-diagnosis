import os
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

ImageFile.LOAD_TRUNCATED_IMAGES = True

class TBImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.encoder = LabelEncoder()
        self.df["label"] = self.encoder.fit_transform(self.df["classes"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        label = self.df.loc[idx, "label"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
