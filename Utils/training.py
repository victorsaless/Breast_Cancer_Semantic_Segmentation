import torch
from Utils.metrics import calculate_metrics
from Utils.logging import Log


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, num_epochs=10, patience=5, min_delta=0.001,
                adjust_lr_after_patience=True, lr_adjust_factor=0.5):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    no_improve_epochs = 0

    log = Log(batch_size=train_loader.batch_size, comment='training')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss, val_dice, val_iou = validate_model(
            model, val_loader, criterion, device, log=log, epoch=epoch
        )
        val_losses.append(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f},"
              f" Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            if adjust_lr_after_patience:
                # Reduz o learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_adjust_factor
                print(f"Reduzindo o learning rate por um fator de "
                      f"{lr_adjust_factor}. Novo valor: "
                      f"{optimizer.param_groups[0]['lr']:.6f}")
                no_improve_epochs = 0
            else:
                print("Interrompendo treinamento devido à falta de melhoria.")
                break

        scheduler.step(val_loss)

    log.close()
    return train_losses, val_losses


def validate_model(
    model, val_loader, criterion, device, num_classes=5, num_samples=2,
    log: Log = None, epoch: int = 0
):
    model.eval()
    val_loss = 0.0
    dice_scores, iou_scores = [], []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            metrics = calculate_metrics(
                outputs, masks, num_classes=num_classes,
                include_background=True
            )

            dice_scores.append(metrics["mean_dice"])
            iou_scores.append(metrics["mean_iou"])

            if idx == 0 and log is not None:
                log.log_tensors_val(images, masks, outputs, epoch)

    mean_dice = sum(dice_scores) / len(dice_scores)
    mean_iou = sum(iou_scores) / len(iou_scores)

    print(
        f"Validation Dice (mean): {mean_dice:.4f}, IoU (mean): {mean_iou:.4f}"
    )

    return val_loss / len(val_loader), mean_dice, mean_iou
