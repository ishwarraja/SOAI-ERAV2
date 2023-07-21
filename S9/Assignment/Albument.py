# Albument.py

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision import datasets

# Define the Cifar10SearchDataset class at the module level
class Cifar10SearchDataset(datasets.CIFAR10):
    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label

def get_albumentations_trainloader(trainset_mean):
    train_transform = A.Compose([
        A.HorizontalFlip(),
        A.ShiftScaleRotate(),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,
                        fill_value=trainset_mean, mask_fill_value=None),
        ToTensorV2(),
    ])

    trainset = Cifar10SearchDataset(root='./data', train=True, download=True, transform=None)
    trainset.transforms = train_transform
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader
