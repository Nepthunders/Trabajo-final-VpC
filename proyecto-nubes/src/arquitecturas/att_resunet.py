import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Bloque residual: dos convoluciones 3x3 más un atajo que suma la entrada a la salida."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
        )
        self.atajo = nn.Identity() if c_in == c_out else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(x) + self.atajo(x))


class AttentionGate(nn.Module):
    """Compuerta de atención (Oktay et al., 2018): la señal del decoder pondera el puente entre 0 y 1."""

    def __init__(self, c_gate, c_skip, c_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(c_gate, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.w_x = nn.Sequential(nn.Conv2d(c_skip, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.psi = nn.Sequential(nn.Conv2d(c_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        return skip * self.psi(self.relu(self.w_g(gate) + self.w_x(skip)))


class AttResUNet(nn.Module):
    """U-Net con bloques residuales y compuertas de atención en los puentes, entrenada desde cero."""

    def __init__(self, classes=4, chans=(32, 64, 128, 256), bottleneck=512):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc = nn.ModuleList()
        c_in = 3
        for c in chans:
            self.enc.append(ResBlock(c_in, c))
            c_in = c
        self.bottleneck = ResBlock(c_in, bottleneck)
        self.up, self.att, self.dec = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c_in = bottleneck
        for c in reversed(chans):
            self.up.append(nn.ConvTranspose2d(c_in, c, 2, stride=2))
            self.att.append(AttentionGate(c, c, c // 2))
            self.dec.append(ResBlock(2 * c, c))
            c_in = c
        self.head = nn.Conv2d(c_in, classes, 1)

    def forward(self, x):
        puentes = []
        for bloque in self.enc:
            x = bloque(x)
            puentes.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, att, dec, puente in zip(self.up, self.att, self.dec, reversed(puentes)):
            x = up(x)
            x = dec(torch.cat([att(x, puente), x], dim=1))
        return self.head(x)


def construir_modelo(preentrenado=True):
    return AttResUNet()
