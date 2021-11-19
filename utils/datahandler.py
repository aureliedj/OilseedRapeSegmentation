from pathlib import Path

from torch.utils.data import DataLoader
from utils.segdataset import SegmentationDataset
from torchvision import transforms as T
from utils.augdata import aug_pipeline
import torch

def get_dataloader_sep_folder(data_dir: str,
                              image_folder: str = 'Images',
                              mask_folder: str = 'Masks',
                              batch_size: int = 2):
    """ Create Train and Test dataloaders from two
        separate Train and Test folders.
        The directory structure should be as follows.
        data_dir
        --Train
        ------Image
        ---------Image1
        ---------ImageN
        ------Mask
        ---------Mask1
        ---------MaskN
        --Test
        ------Image
        ---------Image1
        ---------ImageM
        ------Mask
        ---------Mask1
        ---------MaskM
    Args:
        data_dir (str): The data directory or root.
        image_folder (str, optional): Image folder name. Defaults to 'Image'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Mask'.
        batch_size (int, optional): Batch size of the dataloader. Defaults to 4.
    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """

    transforms = T.Compose([
            T.ToTensor()
        ])

    image_datasets = {
        x: SegmentationDataset(root=Path(data_dir) / x,
                               image_folder=image_folder,
                               mask_folder=mask_folder,
                               transforms = transforms
                               )
        for x in ['Train', 'Test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=8)
        for x in ['Train', 'Test']
    }
    return dataloaders
