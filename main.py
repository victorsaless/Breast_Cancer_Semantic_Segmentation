import cv2
from SGHIST import Dataloader


if __name__ == '__main__':
    image_dir = 'Dataset'
    batch_size = 2
    size = 512
    shuffle = True
    description = True

    dataloader = Dataloader(
        image_dir=image_dir,
        batch_size=batch_size,
        size=size,
        shuffle=shuffle,
        description=description
    )

    train_dataloader = dataloader.get_train()

    for image_batch, mask_batch in train_dataloader:
        image = image_batch[0].detach().cpu().numpy().transpose(1, 2, 0)
        mask = mask_batch[0].detach().cpu().numpy()

        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype('uint8')

        cv2.imshow('Image', image)
        cv2.imshow('Mask', mask * 40)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
