"""
Data Loading Module.

This module defines the PyTorch Lightning DataModule, which encapsulates
all data loading logic (transforms, splitting, and dataloaders) for
training, validation, and testing.
"""

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import datasets
from typing import Optional

# custom imports
from MODEL_config_loader import DataConfiguration, get_transforms_pipeline

class DataLoader_BTI(pl.LightningDataModule):
    def __init__(self, use_augmentation: bool = True, apply_base_transform_to_eval: bool = False):
        """
        Args:
            use_augmentation (bool): If True, applies data augmentation to the training set (defaults to true).
            apply_base_transform_to_eval: If True, applies data augmentation to the evaluation set (defaults to false).
                                          The default for this pipeline its to add the preprocessing pipeline inside
                                          the model forward method, after the training is done.
         """
        super().__init__()
        self.config = DataConfiguration()
        self.use_augmentation = use_augmentation
        self.apply_base_transform_to_eval = apply_base_transform_to_eval

        # Define transforms for each stage
        # Train: Resize -> Pad -> Crop -> Augment -> Normalize
        self.train_transform = get_transforms_pipeline(aug=self.use_augmentation)

        # Val/Test (when enabled): Resize -> Pad -> Crop -> Normalize
        self.val_transform = get_transforms_pipeline(aug=False)


        # Placeholders for datasets
        self.train_dataset: Optional[datasets.ImageFolder] = None
        self.val_dataset: Optional[datasets.ImageFolder] = None
        self.test_dataset: Optional[datasets.ImageFolder] = None
        
    def setup(self, stage: Optional[str] = None):
        """
        Parses data directories and creates dataset objects.
        This method is called automatically by the Lightning Trainer.
        """
        # Stage 'fit' covers both training and validation
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.ImageFolder(
                root=self.config.train_data_root,
                transform=self.train_transform
            )

            self.val_dataset = datasets.ImageFolder(
                root=self.config.val_data_root,
                transform=self.val_transform
            )

        # Stage 'test' handles the evaluation/test set
        if stage == "test" or stage is None:
            transform = self.val_transform if self.apply_base_transform_to_eval else None

            self.test_dataset = datasets.ImageFolder(
                root=self.config.eval_data_root,
                transform=transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            pin_memory=True
        )