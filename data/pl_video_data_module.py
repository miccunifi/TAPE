import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset

class VideoDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule for video datasets

    Args:
        train_dataset (Dataset): Dataset for training
        val_dataset (Dataset): Dataset for validation
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the training data
        num_workers (int): Number of workers for the dataloaders
        pin_memory (bool): Whether to pin memory in the dataloaders
    """
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, batch_size: int, shuffle: bool = True,
                 num_workers: int = 20, pin_memory: bool = True):
        super(VideoDataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)
