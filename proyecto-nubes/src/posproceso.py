"""Posprocesado de las probabilidades: umbral, morfología y área mínima, calibrados en validación."""
import cv2
import numpy as np

from config import AREAS_MIN, IN_H, IN_W, LABELS, METRICA_PRINCIPAL, UMBRALES
from metricas import por_clase

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
VARIANTES = {"A": "A. Umbral 0.5", "B": "B. Umbral por clase", "C": "C. Umbral + área mínima + morfología"}


def morfologia(m):
    """Cierre (rellena huecos) y luego apertura (elimina regiones pequeñas)."""
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, KERNEL)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL).astype(bool)


def aplicar_clase(probs_k, umbral, area_min, morf):
    """probs_k (N, h, w). Devuelve las máscaras booleanas de una clase."""
    pred = probs_k > umbral
    if morf:
        pred = np.stack([morfologia(m) for m in pred])
    if area_min > 0:
        pred[pred.reshape(len(pred), -1).sum(axis=1) < area_min * IN_H * IN_W] = False
    return pred


def aplicar(probs, config, variante="C"):
    """probs (N, 4, h, w). config: diccionario por clase devuelto por calibrar()."""
    salida = []
    for k, label in enumerate(LABELS):
        c = config[label]
        if variante == "A":
            salida.append(aplicar_clase(probs[:, k], 0.5, 0, False))
        elif variante == "B":
            salida.append(aplicar_clase(probs[:, k], c["solo_umbral"], 0, False))
        else:
            salida.append(aplicar_clase(probs[:, k], c["umbral"], c["area_min"], c["morfologia"]))
    return np.stack(salida, axis=1)


def _metrica_clase(inter, p, t, metrica):
    c = np.stack([inter, p, t], axis=-1)[:, None, :]
    return float(por_clase(c)[metrica][0])


def calibrar(probs, targets, metrica=METRICA_PRINCIPAL):
    """Busca en validación, para cada clase, el umbral solo (variante B) y la mejor combinación
    de umbral, morfología y área mínima (variante C), maximizando la métrica principal."""
    config = {}
    for k, label in enumerate(LABELS):
        t_k = targets[:, k]
        t_sum = t_k.reshape(len(t_k), -1).sum(axis=1)
        mejor_b, mejor_c = (-1, None), (-1, None)
        for morf in (False, True):
            for u in UMBRALES:
                base = probs[:, k] > u
                if morf:
                    base = np.stack([morfologia(m) for m in base])
                p_sum = base.reshape(len(base), -1).sum(axis=1)
                inter = (base & t_k).reshape(len(base), -1).sum(axis=1)
                if not morf:
                    valor = _metrica_clase(inter, p_sum, t_sum, metrica)
                    if valor > mejor_b[0]:
                        mejor_b = (valor, u)
                for a in AREAS_MIN:
                    queda = p_sum >= a * IN_H * IN_W
                    valor = _metrica_clase(inter * queda, p_sum * queda, t_sum, metrica)
                    if valor > mejor_c[0]:
                        mejor_c = (valor, {"umbral": u, "area_min": a, "morfologia": morf})
        config[label] = {"solo_umbral": mejor_b[1], **mejor_c[1], "metrica_validacion": round(mejor_c[0], 4)}
    return config
