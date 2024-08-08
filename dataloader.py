import torch
from torch.utils.data import Dataset
import pandas as pd
from cv2 import imread
import pathlib

class ObjectDetectionDataset(Dataset):
    def __init__(self, path: str | pathlib.Path, dataset_type: str, transform=None):
        self.transform = transform

        dataset_path = pathlib.Path(path) # Should eb
        image_path = dataset_path.joinpath("images")
        label_path = dataset_path.joinpath("labels")

        image_files = list(image_path.glob("*"))
        label_files = list(label_path.glob("*"))

        assert len(image_files) == len(label_files), f"Number of images {len(image_files)} and labels {len(label_files)} do not match"


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image_path']
        bounding_boxes = self.data.iloc[idx]['bounding_boxes']

        # Load the image and perform any necessary preprocessing
        image = imread(image_path)
        if self.transform:
            image = self.transform(image)

        # Convert bounding boxes to tensors
        bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)

        return image, bounding_boxes
    

def get_dataloader(dataset_path: str | pathlib.Path, dataset_type: str, batch_size: int = 8, num_workers: int = 4, dataset_percentage: float = 1.0):
    dataset = ObjectDetectionDataset(dataset_path, dataset_type)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
