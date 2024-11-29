import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from SGHIST import Dataloader
from Models.cnn_model import UNetModel
from Utils.training import train_model


if __name__ == '__main__':

    image_dir = 'Dataset'
    batch_size = 4
    size = 512
    shuffle = True
    num_epochs = 20
    patience = 5
    min_delta = 0.001 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = Dataloader(
        image_dir=image_dir,
        batch_size=batch_size,
        size=size,
        shuffle=shuffle
    )
    train_loader = dataloader.get_train()
    val_loader = dataloader.get_val()

    model = UNetModel(in_channels=3, out_channels=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        min_delta=min_delta
    )

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.show()
