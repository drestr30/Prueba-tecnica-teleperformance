# encoding: utf-8
"""
Read images and corresponding labels.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

def load_dataset_from_folder(data_path, transform, batch_size):
    """Loads data from folder of type folder/clases/images.
            data_path: path to the super folder containing folders with class names
            transforms: trnasforms.Compose object containing on-fly image transformation
                        and data augmentation.

            Returns: dataloader to iterate with enumarete(data_loader) or get one batch
                    with next(iter(data_loader))"""

    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader
    

def process_image_file(filepath, back= 'PIL'):

    if back == 'cv':
        img = cv2.imread(filepath)
        # img = cv2.resize(img, (size, size))

    elif back == 'PIL':
        img = Image.open(filepath).convert('RGB')
    return img

def _process_csv_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
    return files

def _process_excel_file(xlsx_path):
    data_frame = pd.read_excel(xlsx_path, sheet_name='Total', index_col=False)
    return data_frame

def get_all_values(d):
    if isinstance(d, dict):
        for v in d.values():
            yield from get_all_values(v)
    elif isinstance(d, list):
        for v in d:
            yield from get_all_values(v)
    else:
        yield d

def plotImages(batch_data, labels=None, title=None, n_images=(4, 4), gray=True ):
    fig, axes = plt.subplots(n_images[0], n_images[1], figsize=(12, 12))
    axes = axes.flatten()
    images = batch_data[0]
    plt.title(title)
    for n, ax in zip(range(n_images[0]*n_images[1]), axes):
        img = images[n].numpy()
        img = np.moveaxis(img, 0, -1)
        if gray:
            ax.imshow(img, cmap='gray') #, vmin=-1, vmax=1)
        else: ax.imshow(img) #, vmin=0, vmax=1) #, vmin=-3, vmax=3)#cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_xticks(())
        ax.set_yticks(())
        # ind = int(np.argmax(batch_data[1][n], axis=-1))
        ind = int(batch_data[1][n])
        ax.set_title((labels[ind]))
    plt.tight_layout()
    plt.show()



def get_transforms(augment=False):

    if augment:

        transformations = transforms.Compose([
            # transforms.ToPILImage(mode='RGB'),
            transforms.ColorJitter(brightness=(0.8,1.2),contrast=(0.8,1.2), saturation=(0.8,1.2)),
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:

        transformations = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    return transformations
    
