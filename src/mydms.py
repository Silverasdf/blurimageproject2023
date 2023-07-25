# My Data Modules - Ryan Peruski, 06/21/2023
# This is a list of types of Data Modules I have used in the past.
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import lightning as L

#This is for image classification
class ImageData(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = 0
        self.num_classes = 2
        self.batch_size = batch_size

    # Makes the actual image datasets
    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders - do data augmentation
        data_transforms = {
            "Training": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.025, 0.025)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "Validation": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.025, 0.025)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "Testing": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.025, 0.025)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.image_datasets = {x: ImageFolder(os.path.join(self.data_dir, x),
                                            data_transforms[x])
                            for x in ["Training", "Validation", "Testing"]}

    #Lightning stuff here
    def train_dataloader(self):
        return DataLoader(self.image_datasets["Training"],
                        batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.image_datasets["Validation"],
                        batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.image_datasets["Testing"],
                        batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    def return_test_filenames(self):
        filenames = self.image_datasets["Testing"].samples
        filenames = [filenames[i][0] for i in range(len(filenames))]
        return filenames
    
#This is the same as the other class, but is only meant for testing. This is used in run_model.py
class ImageDataTest(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = 0
        self.num_classes = 2
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders - do data augmentation
        data_transforms = {
            "Testing": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.025, 0.025)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.image_datasets = ImageFolder(self.data_dir,
                                            data_transforms["Testing"])

    #Lightning stuff here
    def test_dataloader(self):
        return DataLoader(self.image_datasets,
                        batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    def return_test_filenames(self):
        filenames = self.image_datasets.samples
        filenames = [filenames[i][0] for i in range(len(filenames))]
        return filenames