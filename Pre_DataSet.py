import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view


labels = {
    'other': 0,
    'Tumour': 1,
    'Stroma': 2,
    'Lymphocytic infiltrate': 3,
    'Necrosis': 4,
}

colors = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (255, 0, 0),
    4: (0, 255, 255)
}


def filter_image_without_outside_roi(original_image, original_mask):
    return original_image * (original_mask != 0)


def filter_mask_by_cls(original_mask, list_classes):
    new_mask = np.zeros_like(original_mask)
    for cls in list_classes:
        new_mask += cls * (original_mask == cls).astype('uint8')
    return new_mask


def colorizer_mask(original_mask):
    new_mask = np.zeros_like(original_mask, dtype='uint8')
    for key in colors.keys():
        new_mask += np.array(colors[key], dtype='uint8') * (original_mask[:, :, 0] == key)[..., None]
    return new_mask


if __name__ == "__main__":
    size = None
    show = False

    patch_size = (512, 512, 3)
    stride = 512

    dataset_path = 'Dataset/BCSS/'
    split = 'test/'
    images_path = glob(f'{dataset_path}{split}/Image/*')
    masks_path = glob(f'{dataset_path}{split}/Mask/*')

    for path_image, path_mask in tqdm(zip(images_path, masks_path)):
        image, mask = cv2.imread(path_image), cv2.imread(path_mask)
        if size:
            image = cv2.resize(image, size)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        image = filter_image_without_outside_roi(original_image=image, original_mask=mask)
        mask = filter_mask_by_cls(mask, [1, 2, 3, 4])

        print('classes after filter', np.unique(mask))

        patches_image = sliding_window_view(image, patch_size)
        patches_mask = sliding_window_view(mask, patch_size)
        for i in range(0, patches_image.shape[0], stride):
            for j in range(0, patches_image.shape[1], stride):
                patch_mask = patches_mask[i, j, 0]
                patch_image = patches_image[i, j, 0]
                if patch_image.max() == 0:
                    continue
                cv2.imwrite(path_image.replace('BCSS', 'PRE_BCSS').replace('.png', f'{i}{j}.png'), patch_image)
                cv2.imwrite(path_mask.replace('BCSS', 'PRE_BCSS').replace('.png', f'{i}{j}.png'), patch_mask)

        if show:
            mask_show = colorizer_mask(mask)
            cv2.imshow('image', image)
            cv2.imshow('mask', mask_show)
            if cv2.waitKey(0) == ord('q'):
                break

    if show:
        cv2.destroyWindow('image')
        cv2.destroyWindow('mask')
