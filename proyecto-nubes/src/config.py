"""Rutas, constantes y protocolo común. Todo lo que debe ser igual para los cuatro modelos vive aquí."""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EN_KAGGLE = Path("/kaggle/working").exists()

_DATOS_KAGGLE = Path("/kaggle/input/competitions/understanding_cloud_organization")
if "NUBES_DATA" in os.environ:
    DATA_DIR = Path(os.environ["NUBES_DATA"])
elif _DATOS_KAGGLE.exists():
    DATA_DIR = _DATOS_KAGGLE
else:
    DATA_DIR = ROOT / "data"

if "NUBES_PREP" in os.environ:
    PREP_DIR = Path(os.environ["NUBES_PREP"])
elif EN_KAGGLE:
    PREP_DIR = Path("/kaggle/temp/prep")
else:
    PREP_DIR = ROOT / "output" / "prep"

SPLIT_CSV = ROOT / "splits" / "split.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
CHAMPION_JSON = ARTIFACTS_DIR / "champion.json"

LABELS = ["Fish", "Flower", "Gravel", "Sugar"]
COLORES = {"Fish": "#2171b5", "Flower": "#238b45", "Gravel": "#cb181d", "Sugar": "#7b2d8b"}
ORIG_H, ORIG_W = 1400, 2100
IN_H, IN_W = 350, 525
H, W = 352, 544
SEED = 44

MEAN = np.array([0.261, 0.279, 0.327], dtype=np.float32)
STD = np.array([0.242, 0.237, 0.222], dtype=np.float32)

PROTOCOLO = {
    "epocas": int(os.environ.get("NUBES_EPOCAS", 12)),
    "lote": int(os.environ.get("NUBES_LOTE", 16)),
    "lr_max": 1e-3,
    "weight_decay": 1e-4,
    "gamma_focal": 2.0,
    "clip_gradiente": 1.0,
    "workers": int(os.environ.get("NUBES_WORKERS", 2)),
}

METRICA_PRINCIPAL = "dice"

UMBRALES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
AREAS_MIN = [0.0, 0.005, 0.01, 0.02, 0.04, 0.06]

MODELOS = {
    "cnn": "CNN Segmenter",
    "att_resunet": "Attention Residual U-Net",
    "unet_r34": "U-Net + ResNet34",
    "deeplab_r34": "DeepLabV3+ + ResNet34",
}


def carpeta_modelo(nombre):
    ruta = ARTIFACTS_DIR / nombre
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta
