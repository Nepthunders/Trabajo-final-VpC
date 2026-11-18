"""Gráficas compartidas por los notebooks y la app."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from config import COLORES, LABELS


def superponer(img, masks, alpha=0.40):
    """img (h, w, 3) uint8 y masks (4, h, w) booleanas. Devuelve la imagen con las clases coloreadas."""
    salida = img.astype(np.float32) / 255.0
    for k, label in enumerate(LABELS):
        if masks[k].any():
            color = np.array(matplotlib.colors.to_rgb(COLORES[label]))
            salida[masks[k]] = (1 - alpha) * salida[masks[k]] + alpha * color
    return (np.clip(salida, 0, 1) * 255).astype(np.uint8)


def leyenda(fig, ncol=4):
    manijas = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in COLORES.values()]
    fig.legend(manijas, COLORES.keys(), loc="lower center", ncol=ncol)


def curvas(historial, titulo=""):
    """Pérdida, Dice e IoU por época, y Dice de validación por clase."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    e = historial["epoca"]
    axes[0].plot(e, historial["perdida_train"], marker="o", label="Entrenamiento")
    axes[0].plot(e, historial["perdida_val"], marker="o", label="Validación")
    axes[0].set_title("Pérdida (focal + Dice)", fontweight="bold")
    axes[0].legend()
    for col, nombre in (("dice", "Dice global"), ("iou", "IoU global"), ("dice_kaggle", "Dice de la competencia")):
        axes[1].plot(e, historial[col], marker="o", label=nombre)
    axes[1].set_title("Métricas de validación (umbral 0.5)", fontweight="bold")
    axes[1].legend()
    for label in LABELS:
        axes[2].plot(e, historial[f"dice_{label}"], marker="o", color=COLORES[label], label=label)
    axes[2].set_title("Dice de validación por clase", fontweight="bold")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("Época")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    if titulo:
        fig.suptitle(titulo, fontweight="bold")
    plt.tight_layout()
    return fig


def ejemplos(imagenes, targets, preds, dice_img, titulos):
    """Una fila por imagen: etiqueta a la izquierda y predicción a la derecha."""
    n = len(imagenes)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.6 * n))
    axes = np.atleast_2d(axes)
    for fila in range(n):
        for ax, masks, nombre in zip(axes[fila], (targets[fila], preds[fila]), ("Etiqueta", "Predicción")):
            ax.imshow(superponer(imagenes[fila], masks))
            ax.set_title(f"{nombre} | {titulos[fila]} | Dice {dice_img[fila]:.2f}", fontsize=10)
            ax.axis("off")
    leyenda(fig)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def mapas_probabilidad(img, probs, target):
    """Imagen con su etiqueta y el mapa de probabilidad de cada clase."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 3.6))
    axes[0].imshow(superponer(img, target))
    axes[0].set_title("Imagen y etiqueta", fontweight="bold")
    for k, label in enumerate(LABELS):
        im = axes[k + 1].imshow(probs[k].astype(np.float32), vmin=0, vmax=1, cmap="viridis")
        axes[k + 1].contour(target[k], levels=[0.5], colors="white", linewidths=1)
        axes[k + 1].set_title(f"Probabilidad {label}", fontweight="bold")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.012)
    return fig
