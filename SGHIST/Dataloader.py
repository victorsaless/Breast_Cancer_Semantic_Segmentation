import cv2
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, HorizontalFlip, VerticalFlip
from albumentations import RandomRotate90
import os
import torch
from glob import glob


class Data(Dataset):
    def __init__(self, image_dir: str, split: str, transform=None, ext="png"):
        self.image_dir = os.path.join(image_dir, split, "Image")
        self.mask_dir = os.path.join(image_dir, split, "Mask")

        if not os.path.isdir(self.image_dir) or \
           not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(
                f"Diretório {self.image_dir} ou {self.mask_dir} "
                f"não encontrado."
            )

        self.image_paths = sorted(glob(f"{self.image_dir}/*.{ext}"))
        self.mask_paths = sorted(glob(f"{self.mask_dir}/*.{ext}"))

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("O número de imagens e máscaras não corresponde.")

        self.transform = transform or Compose([Resize(256, 256), ToTensorV2()])
        self.mask_transform = Compose([
            Resize(256, 256, interpolation=cv2.INTER_NEAREST)
        ])

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

        image = image.astype('float32') / 255.0
        mask = (mask / 255.0).astype('float32')

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image.to(torch.float32), mask.unsqueeze(0).to(torch.float32)


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
            'train': Compose([
                Resize(self.size, self.size),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                ToTensorV2(),
            ]),
            'eval': Compose([Resize(self.size, self.size), ToTensorV2()]),
            'test': Compose([Resize(self.size, self.size), ToTensorV2()]),
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
