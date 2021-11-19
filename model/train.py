import torch
from pathlib import Path
from utils import datahandler
from utils.model import createDeepLabv3
from utils.trainer import train_model
import torch.optim as optim
from utils.metrics import Pixel_Accuracy, Mean_Intersection_over_Union
if __name__ == "__main__":

    path = r'C:\Users\A60026184\Desktop\ModelB\Data3class'
    data_directory = Path(r"C:\Users\A60026184\Desktop\ModelB\Data3class")
    print('START Loading model...')
    model = createDeepLabv3()
    model = model.train()

    # Create the experiment directory if not present
    exp_directory = Path(r"C:\Users\A60026184\Desktop")
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # Setup the loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    metrics = {'Accuracy': Pixel_Accuracy, 'mIoU': Mean_Intersection_over_Union}
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, (10,25), 0.1)

    print('START DataLoader...')

    # Create the dataloader
    dataloaders = datahandler.get_dataloader_sep_folder(data_directory,
                                            'Images',
                                            'Masks',
                                            2)

    print('START Training...')
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    scheduler,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=20)

    # Save the trained model
    torch.save(model, exp_directory / 'weights.pt')