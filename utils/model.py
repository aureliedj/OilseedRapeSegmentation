""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def createDeepLabv3(outputchannels=3): #2 for vege/novege or 3 for bg/vege/OSR
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    #PATH = r'C:\Users\A60026184\Desktop\deeplabv3_resnet101_coco-586e9e4e.pth'
    print('START Loading model...')
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)

    #auxiliary classifier is removed, and the pretrained weights are frozen.
    #for param in model.parameters():
    #    param.requires_grad = False

    model.classifier = DeepLabHead(2048, outputchannels)

    # Set the model in training mode
    model.train()
    return model