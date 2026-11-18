"""Piezas del entrenamiento que deben ser idénticas para todos: semilla, datos, pérdida, optimizador,
validación por época y guardado de pesos. El ciclo de entrenamiento está escrito en cada notebook."""
import random

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from config import LABELS, METRICA_PRINCIPAL, PROTOCOLO, SEED, carpeta_modelo
from datos import CloudDataset, aumento
from metricas import conteos, por_clase, recortar


def fijar_semilla(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crear_loaders(lote=None, workers=None):
    lote = lote or PROTOCOLO["lote"]
    workers = PROTOCOLO["workers"] if workers is None else workers
    g = torch.Generator().manual_seed(SEED)
    train_dl = DataLoader(CloudDataset("train", aumento()), batch_size=lote, shuffle=True, num_workers=workers,
                          drop_last=True, pin_memory=True, generator=g)
    val_dl = DataLoader(CloudDataset("val"), batch_size=lote * 2, shuffle=False, num_workers=workers, pin_memory=True)
    return train_dl, val_dl


class FocalDice(torch.nn.Module):
    """Pérdida focal (gamma = 2) más pérdida Dice, ambas multi-etiqueta y sobre logits."""

    def __init__(self, gamma=PROTOCOLO["gamma_focal"]):
        super().__init__()
        self.focal = smp.losses.FocalLoss("multilabel", gamma=gamma)
        self.dice = smp.losses.DiceLoss("multilabel", from_logits=True)

    def forward(self, logits, target):
        return self.focal(logits, target) + self.dice(logits, target)


def crear_optimizacion(modelo, pasos_por_epoca, epocas=None):
    """AdamW con OneCycleLR (sube y luego baja la tasa de aprendizaje) y escalador para precisión mixta."""
    epocas = epocas or PROTOCOLO["epocas"]
    opt = torch.optim.AdamW(modelo.parameters(), lr=PROTOCOLO["lr_max"] / 10, weight_decay=PROTOCOLO["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=PROTOCOLO["lr_max"], total_steps=epocas * pasos_por_epoca)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    return opt, sched, scaler


@torch.no_grad()
def validar(modelo, loader, criterio, device):
    """Pérdida y métricas de validación con umbral 0.5, sobre 350x525."""
    modelo.eval()
    perdida, n, cnt = 0.0, 0, []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = modelo(x)
        perdida += criterio(logits.float(), y).item() * len(x)
        n += len(x)
        pred = recortar((torch.sigmoid(logits.float()) > 0.5).cpu().numpy())
        cnt.append(conteos(pred, recortar(y.cpu().numpy() > 0.5)))
    m = por_clase(np.concatenate(cnt))
    fila = {"perdida_val": perdida / n, "dice": m["dice"].mean(), "iou": m["iou"].mean(),
            "dice_kaggle": m["dice_kaggle"].mean()}
    fila.update({f"dice_{l}": d for l, d in zip(LABELS, m["dice"])})
    return fila


def estado_cpu(modelo):
    return {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}


def guardar_pesos(estado, nombre, meta):
    """Guarda los pesos en media precisión (fp16), la mitad de tamaño que en fp32."""
    ruta = carpeta_modelo(nombre) / "pesos_fp16.pt"
    estado16 = {k: (v.half() if v.is_floating_point() else v) for k, v in estado.items()}
    torch.save({"modelo": nombre, "state_dict": estado16, **meta}, ruta)
    print(f"Pesos guardados en {ruta} ({ruta.stat().st_size / 1e6:.1f} MB)")
    return ruta


def cargar_pesos(modelo, nombre, device=None):
    """Carga los pesos fp16 en un modelo float32. Devuelve el diccionario guardado (sin pesos)."""
    device = device or dispositivo()
    ckpt = torch.load(carpeta_modelo(nombre) / "pesos_fp16.pt", map_location="cpu", weights_only=False)
    modelo.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ckpt["state_dict"].items()})
    modelo.to(device).eval()
    return {k: v for k, v in ckpt.items() if k != "state_dict"}


def metrica_seleccion(fila):
    return fila[METRICA_PRINCIPAL]
