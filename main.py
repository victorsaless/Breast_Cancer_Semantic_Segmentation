import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from SGHIST import Dataloader
from Models.cnn_model import UNetModel 


def train_model(model, dataloader, criterion, optimizer, num_epochs=20):
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        train_losses.append(epoch_loss)
 
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return train_losses


if __name__ == '__main__':
    image_dir = 'Dataset'
    batch_size = 2
    size = 512
    shuffle = True
    description = True
    split = 'train'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = Dataloader(
        image_dir=image_dir,
        batch_size=batch_size,
        size=size,
        shuffle=shuffle,
        description=description,
        split=split
    )

    train_dataloader = dataloader.get_train()

    model = UNetModel(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = train_model(
        model, train_dataloader, criterion, optimizer, num_epochs=10
    )

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
