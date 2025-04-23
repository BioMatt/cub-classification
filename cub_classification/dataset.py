import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms as T

#torch.set_float32_matmul_precision("medium") # Set precision to be relatively low to speed up computations
# The line above is commented out because precision='16-mixed' was set in pl.trainer in the train.py script

class CUBDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        super().__init__() # Set up the super class
        self.csv_file = csv_file
        self.data_dir = data_dir
        self.transform = transform

        self.samples = []
        with open(self.csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        image_filename = Path(row["filename"])
        img_path = self.data_dir / "images" / image_filename

        # Load image
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        class_id = int(row["class_id"]) - 1 # Using -1 for a zero index, because the model predicts class 0 by default

        x_min = float(row["x_min"])
        y_min = float(row["y_min"])
        x_max = float(row["x_max"])
        y_max = float(row["y_max"])

        bounding_box = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        return img, (class_id, bounding_box)
    
class CUBDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        if transform is None:
            transform = T.ToTensor()
        self.transform = transform

    def setup(self, stage=None):
        self.train_dataset = CUBDataset(
            csv_file = self.data_dir / "train.csv",
            data_dir = self.data_dir,
            transform = self.transform
        )

        self.val_dataset = CUBDataset(
            csv_file = self.data_dir / "val.csv",
            data_dir = self.data_dir,
            transform = self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True, # Randomize the order of the input so the NN does not just learn that
            num_workers=15
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers=15
        )