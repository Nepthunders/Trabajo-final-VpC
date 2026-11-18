import segmentation_models_pytorch as smp


def construir_modelo(preentrenado=True):
    """U-Net con encoder ResNet34 preentrenado en ImageNet. El decoder recibe un puente
    desde cada nivel del encoder y termina en 4 canales, uno por clase."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if preentrenado else None,
        classes=4,
        activation=None,
    )
