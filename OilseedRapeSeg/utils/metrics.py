
"""
Source code : https://github.com/giovanniguidi/deeplabV3-PyTorch/blob/master/trainers/trainer.py
"""

import numpy as np
### ---------- MODEL A ---------- ###
#num_class=2
num_class = 3
confusion_matrix = np.zeros((num_class,)*2)

def Pixel_Accuracy(confusion_matrix):
    Acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return Acc

def Pixel_Accuracy_Class(confusion_matrix):
    Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    #Acc = np.nanmean(Acc)
    return Acc

def Mean_Intersection_over_Union(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU

def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))

    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def _generate_matrix(num_class, gt_image, pre_image):
    mask = (gt_image >= 0) & (gt_image < num_class)
    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    count = np.bincount(label, minlength=num_class**2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix

def reset(num_class):
    return np.zeros((num_class,) * 2)