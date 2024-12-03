import albumentations as A 
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from albumentations.pytorch import ToTensorV2
from glob import glob
import numpy as np
import torch
import cv2
import os


class Data(Dataset):
    def __init__(self, image_dir: str, split: str, transform=None, ext="png"):
        self.image_dir = os.path.join(image_dir, split, "Image")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f"Diretório {self.image_dir} "
                f"não encontrado."
            )
        self.transform = transform

        self.image_paths = glob(f"{self.image_dir}/*.{ext}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_paths[idx]
        mask_path = image_path.replace("Image", "Mask")

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Dimensões de {image_path} e {mask_path} não correspondem."
            )

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask


class MyPreProcessing(A.BasicTransform):
    def __init__(self, num_classes=5, always_apply=False, p=1.0):
        super(MyPreProcessing, self).__init__(always_apply, p)
        self.num_classes = num_classes

    def apply(self, image, shape, **kwargs):
        return (image / 255.).astype('float32')

    def apply_to_mask(self, mask, **kwargs):
        return np.eye(self.num_classes)[mask].astype('float32').transpose(2, 0, 1)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}


class Dataloader:
    def __init__(self, image_dir: str, batch_size: int, size: int,
                 shuffle: bool = True, subset: int = 0):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.size = size
        self.shuffle = shuffle
        self.subset = subset
        self.transform = self.compose()

    def compose(self):
        return {
            'train': A.Compose([
                A.HorizontalFlip(p=0.15),
                A.VerticalFlip(p=0.15),
                A.RandomRotate90(p=0.15),
                MyPreProcessing(),
                ToTensorV2(),
            ]),
            'eval': A.Compose([MyPreProcessing(), ToTensorV2()]),
            'test': A.Compose([MyPreProcessing(), ToTensorV2()]),
        }

    def get_dataloader(self, split: str) -> DataLoader:
        dataset = Data(self.image_dir, split, self.transform[split])

        if self.subset > 0:
            indices = torch.randperm(len(dataset))[:self.subset]
            sampler = SubsetRandomSampler(indices)
            dataLoader = DataLoader(
                dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            dataLoader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=self.shuffle
            )

        return dataLoader

    def get_train(self) -> DataLoader:
        return self.get_dataloader('train')

    def get_val(self) -> DataLoader:
        return self.get_dataloader('eval')

    def get_test(self) -> DataLoader:
        return self.get_dataloader('test')
