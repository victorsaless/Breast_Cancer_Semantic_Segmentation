import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from SGHIST import Dataloader
from Models.cnn_model import UNetModel
from Utils.training import train_model, validate_model, test_model
from Utils.logging import Log


if __name__ == '__main__':
    image_dir = 'Dataset/PRE_BCSS/'
    batch_size = 4
    size = 512
    shuffle = True
    num_epochs = 20
    patience = 5
    min_delta = 0.001
    num_classes = 5
    subset_size = 500
    log_comment = "UNet_Training"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataloader = Dataloader(
        image_dir=image_dir,
        batch_size=batch_size,
        size=size,
        shuffle=shuffle,
        subset=subset_size
    )

    train_loader = dataloader.get_train()
    val_loader = dataloader.get_val()
    test_loader = dataloader.get_test()

    model = UNetModel(in_channels=3, out_channels=num_classes).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, cooldown=2
    )

    logger = Log(batch_size=batch_size, comment=log_comment)

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        min_delta=min_delta,
    )

    for epoch in range(len(train_losses)):
        logger.log_scalar_train(train_losses[epoch], epoch, scalar_name="Loss")
        logger.log_scalar_val(val_losses[epoch], epoch, scalar_name="Loss")
        logger.log_scalar_hiper(scheduler.optimizer.param_groups[0]['lr'], epoch, scalar_name="Learning Rate")

    val_metrics = validate_model(
        model, val_loader, criterion, device, num_samples=3
    )
    print(
        f"Validation Metrics: Dice = {val_metrics[1]}, IoU = {val_metrics[2]}"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    print("\n=== Testing the Model on the Test Suitee ===")
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_dice, test_iou = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes
    )
    print(
        f"Test Results:\n"
        f"Test Loss: {test_loss:.4f}\n"
        f"Test Dice: {test_dice:.4f}\n"
        f"Test IoU: {test_iou:.4f}"
    )

    logger.close()
    print("Training and testing completed.")
