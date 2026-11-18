"""Métricas a partir de conteos por imagen y clase. Todas se calculan sobre 350x525, sin el relleno.

Conteos: arreglo (N, 4, 3) con, para cada imagen y clase, los píxeles de la intersección,
los píxeles predichos y los píxeles de la etiqueta. Con eso se obtiene cualquier métrica
sin volver a cargar las máscaras, y se puede hacer bootstrap sobre las imágenes.
"""
import numpy as np
import pandas as pd

from config import IN_H, IN_W, LABELS

NOMBRES_METRICAS = {
    "dice": "Dice global",
    "iou": "IoU global",
    "dice_kaggle": "Dice de la competencia",
    "precision_pixel": "Precisión por píxel",
    "recall_pixel": "Recall por píxel",
    "precision_imagen": "Precisión por imagen",
    "recall_imagen": "Recall por imagen",
}


def recortar(arr):
    """Quita el relleno: (..., 352, 544) a (..., 350, 525)."""
    return arr[..., :IN_H, :IN_W]


def conteos(pred, target):
    """pred y target booleanos (N, 4, h, w) ya recortados."""
    inter = (pred & target).sum(axis=(2, 3))
    return np.stack([inter, pred.sum(axis=(2, 3)), target.sum(axis=(2, 3))], axis=-1).astype(np.int64)


def por_clase(c):
    """Diccionario métrica -> arreglo de 4 valores (uno por clase)."""
    inter, p, t = c[..., 0], c[..., 1], c[..., 2]
    si, sp, st = inter.sum(0).astype(float), p.sum(0).astype(float), t.sum(0).astype(float)
    total = p + t
    dice_img = np.where(total == 0, 1.0, 2 * inter / np.maximum(total, 1))
    pres_p, pres_t = p > 0, t > 0
    tp = (pres_p & pres_t).sum(0)
    return {
        "dice": 2 * si / np.maximum(sp + st, 1),
        "iou": si / np.maximum(sp + st - si, 1),
        "dice_kaggle": dice_img.mean(0),
        "precision_pixel": si / np.maximum(sp, 1),
        "recall_pixel": si / np.maximum(st, 1),
        "precision_imagen": tp / np.maximum(pres_p.sum(0), 1),
        "recall_imagen": tp / np.maximum(pres_t.sum(0), 1),
    }


def promedio(c, metrica):
    return float(por_clase(c)[metrica].mean())


def tabla(c):
    """DataFrame con una fila por métrica y columnas por clase más el promedio."""
    m = por_clase(c)
    df = pd.DataFrame({k: v for k, v in m.items()}, index=LABELS).T
    df["Promedio"] = df.mean(axis=1)
    return df


def dice_por_imagen(c):
    """Dice promedio de las 4 clases en cada imagen (1 en una clase vacía bien predicha)."""
    inter, p, t = c[..., 0], c[..., 1], c[..., 2]
    total = p + t
    return np.where(total == 0, 1.0, 2 * inter / np.maximum(total, 1)).mean(axis=1)


def bootstrap(conteos_por_modelo, metrica, n=1000, seed=44):
    """Remuestrea imágenes con reemplazo (las mismas para todos los modelos) y recalcula la métrica.

    Devuelve un DataFrame (n, modelos) con la métrica promedio de las 4 clases en cada réplica.
    """
    rng = np.random.default_rng(seed)
    nombres = list(conteos_por_modelo)
    N = len(next(iter(conteos_por_modelo.values())))
    filas = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        filas.append({m: promedio(conteos_por_modelo[m][idx], metrica) for m in nombres})
    return pd.DataFrame(filas)
