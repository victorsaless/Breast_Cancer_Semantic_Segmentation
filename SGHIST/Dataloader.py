from glob import glob
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import cv2
import os
import torch
from tqdm import tqdm


class Data(Dataset):
    def __init__(self, image_dir: str, split: str, transform=None, ext="png"):
        self.image_dir = os.path.join(image_dir, split, "Image")
        self.mask_dir = os.path.join(image_dir, split, "Mask")

        if not os.path.isdir(self.image_dir) or \
           not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(
                (f"Diretório(s) {self.image_dir} ou {self.mask_dir} "
                 "não encontrado(s).")
            )

        self.image_paths = sorted(glob(f"{self.image_dir}/*.{ext}"))
        self.mask_paths = sorted(glob(f"{self.mask_dir}/*.{ext}"))

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Número de imagens e máscaras não corresponde.")

        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (256, 256),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = self.transform(image)
        mask = self.mask_transform(mask).squeeze().type(torch.long)

        return image, mask


class Dataloader:
    def __init__(self, split: str, batch_size: int, shuffle: bool,
                 image_dir: str = "Dataset", subset: int = 0,
                 description: bool = False, size: int = 256,
                 transform=None) -> None:
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subset = subset
        self.description = description
        self.size = size
        self.transform = transform or self.compose()

    def compose(self):
        trans_test = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7, 0.5, 0.7], std=[0.2, 0.2, 0.2])
        ])

        trans_train = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7, 0.5, 0.7], std=[0.2, 0.2, 0.2])
        ])

        return {
            'train': trans_train,
            'eval': trans_test,
            'test': trans_test
        }

    def get_dataloader(self, split: str) -> DataLoader:
        dataset = Data(self.image_dir, split, self.transform[split])

        if self.subset > 0:
            indices = torch.randperm(len(dataset))[:self.subset]
            sampler = SubsetRandomSampler(indices)
            dataLoader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler
            )
        else:
            dataLoader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle
            )

        if self.description:
            return tqdm(dataLoader, desc=f"Loading {split} data")
        return dataLoader

    def get_train(self) -> DataLoader:
        return self.get_dataloader('train')

    def get_val(self) -> DataLoader:
        return self.get_dataloader('val')

    def get_test(self) -> DataLoader:
        return self.get_dataloader('test')
