import torch
from Utils.metrics import calculate_metrics
from Utils.visualization import visualize_results


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, num_epochs=20, patience=5, min_delta=0.001):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    no_improve_epochs = 0

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
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Dice: {val_dice:.4f} | IoU: {val_iou:.4f}"
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping triggered."
                  "(Não houve melhora dentro do intervalo de paciência)")
            break

        scheduler.step(val_loss)

    return train_losses, val_losses


def validate_model(model, val_loader, criterion, device, num_samples=2):
    model.eval()
    val_loss = 0.0
    dice_scores, iou_scores = [], []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            dice, iou = calculate_metrics(outputs, masks)
            dice_scores.append(dice)
            iou_scores.append(iou)

            if idx == 0:
                visualize_results(images, masks, outputs, num_samples)

    return (
        val_loss / len(val_loader),
        sum(dice_scores) / len(dice_scores),
        sum(iou_scores) / len(iou_scores),
    )
