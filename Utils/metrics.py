import torch


def calculate_dice(preds, masks, num_classes, include_background=True):

    dice_scores = []
    for cls in range(num_classes):
        if not include_background and cls == 0:
            continue

        pred = preds[:, cls]
        mask = masks[:, cls]

        intersection = (pred * mask).sum()
        union = pred.sum() + mask.sum()

        dice = (2 * intersection) / (union + 1e-7)
        dice_scores.append(dice.item())

    return dice_scores


def calculate_iou(preds, masks, num_classes, include_background=True):

    iou_scores = []
    for cls in range(num_classes):
        if not include_background and cls == 0:
            continue

        pred = preds[:, cls]
        mask = masks[:, cls]

        intersection = (pred * mask).sum()
        union = pred.sum() + mask.sum() - intersection

        iou = intersection / (union + 1e-7)
        iou_scores.append(iou.item())

    return iou_scores


def calculate_metrics(
    outputs, masks, num_classes, threshold=0.5, include_background=True
):

    outputs = torch.sigmoid(outputs)
    preds = (outputs > threshold).float()
    masks = (masks > threshold).float()

    dice_scores = calculate_dice(preds, masks, num_classes, include_background)
    iou_scores = calculate_iou(preds, masks, num_classes, include_background)

    return {
        "dice_per_class": dice_scores,
        "iou_per_class": iou_scores,
        "mean_dice": sum(dice_scores) / len(dice_scores),
        "mean_iou": sum(iou_scores) / len(iou_scores),
    }
