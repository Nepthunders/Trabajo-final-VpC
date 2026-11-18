import torch.nn as nn


class ConvBlock(nn.Module):
    """Dos convoluciones 3x3 con BatchNorm y ReLU, y Dropout espacial al final."""

    def __init__(self, c_in, c_out, drop):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
        )

    def forward(self, x):
        return self.block(x)


class CNNSegmenter(nn.Module):
    """Encoder-decoder de 4 niveles SIN puentes (skip connections), entrenado desde cero."""

    def __init__(self, classes=4, chans=(32, 64, 128, 256), bottleneck=512):
        super().__init__()
        drops = (0.05, 0.10, 0.15, 0.20)
        enc, c_in = [], 3
        for c, d in zip(chans, drops):
            enc += [ConvBlock(c_in, c, d), nn.MaxPool2d(2)]
            c_in = c
        self.encoder = nn.Sequential(*enc)
        self.bottleneck = ConvBlock(c_in, bottleneck, 0.20)
        dec, c_in = [], bottleneck
        for c, d in zip(reversed(chans), reversed(drops)):
            dec += [nn.ConvTranspose2d(c_in, c, 2, stride=2), ConvBlock(c, c, d)]
            c_in = c
        self.decoder = nn.Sequential(*dec)
        self.head = nn.Conv2d(c_in, classes, 1)

    def forward(self, x):
        return self.head(self.decoder(self.bottleneck(self.encoder(x))))


def construir_modelo(preentrenado=True):
    return CNNSegmenter()
