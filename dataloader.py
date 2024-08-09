import torch
from torch.utils.data import Dataset
import pandas as pd
from cv2 import imread
import pathlib
import torch.utils.data

from torchvision import models, datasets, tv_tensors
from torchvision.transforms import v2

torch.manual_seed(0)


class ObjectDetectionDataset(Dataset):
    def __init__(self, path: str | pathlib.Path, dataset_type: str, transform=None):
        self.transform = transform

        dataset_path = pathlib.Path(path).joinpath(dataset_type)
        image_path = dataset_path.joinpath("images")
        label_path = dataset_path.joinpath("labels")
        
        self.dataset_type = dataset_type

        self.image_files = list(image_path.glob("*"))
        self.label_files = list(label_path.glob("*"))
        
        if len(self.image_files) != len(self.label_files):
            if len(self.image_files) < len(self.label_files):
                print(f"{self.dataset_type}: Number of images {len(self.image_files)} and labels {len(self.label_files)} do not match, removing extra labels")
                self.label_files = []
                for image_file in self.image_files:
                    image_name = image_file.stem
                    self.label_files.append(label_path.joinpath(image_name + ".csv"))
            else:
                raise NotImplementedError("Number of labels cannot be greater than number of images")
                    
        
        sorted(self.image_files)
        sorted(self.label_files)
        
        
        assert len(self.image_files) == len(self.label_files), f"Number of images {len(self.image_files)} and labels {len(self.label_files)} do not match"


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        print(image_path)
        print(label_path)
        print(type(image_path))
        print(type(label_path))
        exit("")
        
        image = imread(image_path)
        label = pd.read_csv(label_path)
        print(image.shape)

        # Load the image and perform any necessary preprocessing
        
        print(label)
        exit("")

        if self.transform:
            image = self.transform(image)

        # Convert bounding boxes to tensors
        bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)

        return image, bounding_boxes
    

def get_dataloader(dataset_path: str | pathlib.Path, dataset_type: str, batch_size: int = 8, num_workers: int = 0, dataset_percentage: float = 1.0):
    dataset = ObjectDetectionDataset(dataset_path, dataset_type)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True if dataset_type == "train" else False, num_workers=num_workers)
    return dataloader
