import matplotlib.pyplot as plt
import torch


def visualize_results(images, masks, preds, num_samples=2):
    images = images.cpu()
    masks = masks.cpu()
    preds = (torch.sigmoid(preds) > 0.5).float().cpu()

    for i in range(num_samples):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(images[i].permute(1, 2, 0).numpy().clip(0, 1))
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(masks[i][0], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(preds[i][0], cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.show()
