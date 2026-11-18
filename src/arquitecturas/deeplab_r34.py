import segmentation_models_pytorch as smp


def construir_modelo(preentrenado=True):
    """DeepLabV3+ con el mismo encoder ResNet34 que la U-Net, para comparar solo el decoder.
    Reducción x8: con 352x544 el mapa del ASPP mide 44x68 y las tasas 6, 12 y 18 caben en él."""
    return smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet" if preentrenado else None,
        classes=4,
        activation=None,
        encoder_output_stride=8,
        decoder_atrous_rates=(6, 12, 18),
    )
