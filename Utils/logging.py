
import datetime
import git
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class Stamp:
    def __init__(
            self, repo_path: str = "C:/Users/sales/Documents/mestrado/Breast_Cancer_Semantic_Segmentation"
            ) -> None:
        self._repo = git.Repo(repo_path)
        self._commit = self._repo.head.commit
        self._time = datetime.datetime.now().strftime("%d%m%Y%H%M")
        self._max_commit_character = 8

    def timestamp(self):
        return self._time

    def get_hex(self):
        return self._commit.hexsha[:self._max_commit_character]

    def get_details(self):
        return f'message: {self._commit.message} | autor:{self._commit.author} | Data do commit: {self._commit.committed_datetime}'


class Log:
    def __init__(self,  batch_size: int, comment: str = '', path: str = 'runs/') -> None:
        self.stamp = Stamp()
        self.batch_size = batch_size
        self.writer = SummaryWriter(
            log_dir=f'{path}{self.stamp.get_hex()}{self.stamp.timestamp()}{comment}',
            comment=f'{self.stamp.get_details()}',
            filename_suffix=f'{self.stamp.timestamp()}')
        self.model_saved = False

    def _log_scalar(self, scalar: float, epoch: int, path: str, mean: bool = True) -> None:
        self.writer.add_scalar(path, np.mean(scalar) if mean else scalar, epoch)
        self.writer.flush()

    def log_scalar_train(self, scalar, epoch, scalar_name='IOU', mean: bool = True):
        self._log_scalar(scalar=scalar, epoch=epoch, path=f'{scalar_name}/Train', mean=mean)

    def log_scalar_val(self, scalar, epoch, scalar_name='IOU', mean: bool = True):
        self._log_scalar(scalar=scalar, epoch=epoch, path=f'{scalar_name}/Val', mean=mean)

    def log_scalar_hiper(self, scalar, epoch, scalar_name='LR'):
        self._log_scalar(scalar=scalar, epoch=epoch, path=f'HIPER/{scalar_name}', mean=False)

    def log_image(self, images, epoch, path: str = None):
        img_grid = make_grid(images, nrow=self.batch_size)
        self.writer.add_image(path, img_grid, global_step=epoch)

    def log_tensors(self, image, mask, output, epoch: int, split: str):
        image = (image.detach().cpu() * 255).to(torch.uint8)
        mask = mask.detach().cpu()
        output = (output.detach().cpu() > 0.5) * 1.0

        mask_image = torch.zeros(mask.shape[0], 3, mask.shape[2], mask.shape[3], dtype=torch.uint8)
        output_image = torch.zeros(output.shape[0], 3, output.shape[2], output.shape[3], dtype=torch.uint8)
        for i in range(5):
            color = torch.randint(0, 255, (1, 3, 1, 1), dtype=torch.uint8)
            mask_image += mask[:, i, :, :][:, None, :, :].repeat(1, 3, 1, 1).to(torch.uint8) * color
            output_image += output[:, i, :, :][:, None, :, :].repeat(1, 3, 1, 1).to(torch.uint8) * color
        mask_image.clamp(0, 255)
        output_image.clamp(0, 255)
        images = torch.concat([image, mask_image, output_image], dim=0)
        self.log_image(images, path=f'tensors/{split}', epoch=epoch)

    def log_tensors_train(self, image, mask, output, epoch: int):
        self.log_tensors(image, mask, output, epoch, 'train')

    def log_tensors_val(self, image, mask, output, epoch: int):
        self.log_tensors(image, mask, output, epoch, 'val')

    def close(self):
        self.writer.close()

    def log_model(self, model, images_input, forced_log: bool = False):
        if not self.model_saved or forced_log:
            print('Log Model')
            self.writer.add_graph(model, images_input)
            self.model_saved = True

    def log_embedding(self, features, class_labels, labels):
        self.writer.add_embedding(features, metadata=class_labels, label_img=labels)
