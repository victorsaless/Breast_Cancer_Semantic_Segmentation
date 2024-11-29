import torch


def calculate_dice(preds, masks):
    intersection = (preds * masks).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
    dice = (2.0 * intersection) / (union + 1e-7)
    return dice.mean().item()


def calculate_iou(preds, masks):
    intersection = (preds * masks).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) - intersection
    iou = intersection / (union + 1e-7)
    return iou.mean().item()


def calculate_metrics(outputs, masks, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    preds = (outputs > threshold).float()
    masks = (masks > threshold).float()

    dice = calculate_dice(preds, masks)
    iou = calculate_iou(preds, masks)

    return dice, iou
