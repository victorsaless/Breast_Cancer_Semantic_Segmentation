import segmentation_models_pytorch as smp


def UNetModel(in_channels=3, out_channels=5, encoder_name="resnet34"):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_channels,
        activation="sigmoid"
    )

    return model
