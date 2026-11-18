"""Lectura de etiquetas, preparación única de imágenes y máscaras, Dataset y aumento de datos."""
from concurrent.futures import ProcessPoolExecutor

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (DATA_DIR, H, IN_H, IN_W, LABELS, MEAN, ORIG_H, ORIG_W, PREP_DIR, SEED,
                    SPLIT_CSV, STD, W)


def rle_to_mask(rle, height=ORIG_H, width=ORIG_W):
    """RLE de Kaggle (índice desde 1, recorrido por columnas) a máscara binaria (height, width)."""
    if not isinstance(rle, str) or rle == "":
        return np.zeros((height, width), dtype=np.uint8)
    nums = np.array(rle.split(), dtype=np.int64)
    starts, lengths = nums[0::2] - 1, nums[1::2]
    delta = np.zeros(height * width + 1, dtype=np.int32)
    np.add.at(delta, starts, 1)
    np.add.at(delta, starts + lengths, -1)
    return (np.cumsum(delta[:-1]) > 0).astype(np.uint8).reshape(width, height).T


def mask_to_rle(mask):
    """Máscara binaria (height, width) a RLE de Kaggle."""
    pixels = np.concatenate([[0], mask.T.flatten().astype(np.uint8), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(map(str, runs))


def cargar_anotaciones():
    """Una fila por imagen y una columna por clase con el RLE ('' si la clase no está)."""
    df = pd.read_csv(DATA_DIR / "train.csv")
    df[["Image", "Label"]] = df["Image_Label"].str.rsplit("_", n=1, expand=True)
    return df.pivot(index="Image", columns="Label", values="EncodedPixels")[LABELS].fillna("")


def crear_split(presencia):
    """80/10/10 estratificado por combinación de etiquetas, con semilla fija."""
    patron = presencia.astype(str).agg("".join, axis=1)
    split = pd.Series("train", index=presencia.index, name="split")
    resto = patron.groupby(patron).sample(frac=0.20, random_state=SEED).index
    test = patron[resto].groupby(patron[resto]).sample(frac=0.50, random_state=SEED).index
    split[resto] = "val"
    split[test] = "test"
    return split


def preprocesar_imagen(img_rgb):
    """Reduce a 350x525 y rellena con negro hasta 352x544 (abajo y a la derecha)."""
    lienzo = np.zeros((H, W, 3), dtype=np.uint8)
    lienzo[:IN_H, :IN_W] = cv2.resize(img_rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    return lienzo


def normalizar(img_uint8):
    """(H, W, 3) uint8 a tensor (3, H, W) normalizado con la media y desviación del EDA."""
    x = (img_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def _procesar(args):
    nombre, rles = args
    img = cv2.cvtColor(cv2.imread(str(DATA_DIR / "train_images" / nombre), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = np.zeros((H, W, len(LABELS)), dtype=np.uint8)
    for k, rle in enumerate(rles):
        if rle:
            full = rle_to_mask(rle).astype(np.float32)
            mask[:IN_H, :IN_W, k] = cv2.resize(full, (IN_W, IN_H), interpolation=cv2.INTER_AREA) > 0.5
    return preprocesar_imagen(img), mask


def preparar_datos(forzar=False):
    """Reduce una sola vez todas las imágenes y máscaras y las guarda en PREP_DIR con su partición."""
    indice_csv = PREP_DIR / "index.csv"
    if indice_csv.exists() and not forzar:
        indice = pd.read_csv(indice_csv)
        print(f"Datos ya preparados en {PREP_DIR}")
        print(indice["split"].value_counts().to_string())
        return indice

    pivot = cargar_anotaciones()
    presencia = (pivot != "").astype(int)
    if SPLIT_CSV.exists():
        split = pd.read_csv(SPLIT_CSV, index_col=0)["split"].reindex(pivot.index)
        print(f"Usando la partición existente: {SPLIT_CSV}")
    else:
        split = crear_split(presencia)
        SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(SPLIT_CSV)
        print(f"Partición creada: {SPLIT_CSV}")
    print(split.value_counts().to_string(), flush=True)

    PREP_DIR.mkdir(parents=True, exist_ok=True)
    n = len(pivot)
    images = np.lib.format.open_memmap(PREP_DIR / "images.npy", mode="w+", dtype=np.uint8, shape=(n, H, W, 3))
    masks = np.lib.format.open_memmap(PREP_DIR / "masks.npy", mode="w+", dtype=np.uint8, shape=(n, H, W, len(LABELS)))
    tareas = [(nombre, list(pivot.loc[nombre])) for nombre in pivot.index]
    with ProcessPoolExecutor() as ex:
        for i, (img, mask) in enumerate(ex.map(_procesar, tareas, chunksize=16)):
            images[i], masks[i] = img, mask
            if (i + 1) % 500 == 0 or i + 1 == n:
                print(f"{i + 1}/{n}", flush=True)
    images.flush()
    masks.flush()

    indice = pd.DataFrame({"Image": pivot.index, "pos": np.arange(n), "split": split.values})
    indice = pd.concat([indice, presencia.reset_index(drop=True)], axis=1)
    indice.to_csv(indice_csv, index=False)
    print(f"Listo: {PREP_DIR}")
    return indice


def cargar_indice(split=None):
    indice = pd.read_csv(PREP_DIR / "index.csv")
    return indice if split is None else indice[indice["split"] == split].reset_index(drop=True)


def aumento():
    """Flips, transformación afín suave y cambios leves de brillo y contraste. No altera la textura."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    ])


class CloudDataset(Dataset):
    """Lee las imágenes y máscaras ya reducidas. Devuelve (3, H, W) normalizada y (4, H, W) en {0, 1}."""

    def __init__(self, split, transform=None):
        self.filas = cargar_indice(split)
        self.transform = transform
        self.images = self.masks = None

    def _abrir(self):
        self.images = np.load(PREP_DIR / "images.npy", mmap_mode="r")
        self.masks = np.load(PREP_DIR / "masks.npy", mmap_mode="r")

    def __len__(self):
        return len(self.filas)

    def imagen_y_mascara(self, i):
        if self.images is None:
            self._abrir()
        pos = int(self.filas.loc[i, "pos"])
        return np.array(self.images[pos]), np.array(self.masks[pos])

    def __getitem__(self, i):
        image, mask = self.imagen_y_mascara(i)
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return normalizar(image), torch.from_numpy(np.ascontiguousarray(mask.transpose(2, 0, 1))).float()
