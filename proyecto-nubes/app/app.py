"""App de despliegue: segmenta patrones de nubes en imágenes satelitales nuevas con el modelo campeón.

Uso, desde la raíz del repo:
    streamlit run app/app.py
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arquitecturas import construir
from config import ARTIFACTS_DIR, CHAMPION_JSON, COLORES, IN_H, IN_W, LABELS, MODELOS
from datos import mask_to_rle, normalizar, preprocesar_imagen
from entrenamiento import cargar_pesos
from graficas import superponer
from posproceso import aplicar

st.set_page_config(page_title="Segmentación de nubes", layout="wide")


def modelos_disponibles():
    return [m for m in MODELOS if (ARTIFACTS_DIR / m / "pesos_fp16.pt").exists()
            and (ARTIFACTS_DIR / m / "posproceso.json").exists()]


def leer_json(ruta):
    return json.loads(ruta.read_text()) if ruta.exists() else {}


@st.cache_resource
def cargar_modelo(nombre):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = construir(nombre, preentrenado=False)
    meta = cargar_pesos(modelo, nombre, device)
    config_post = leer_json(ARTIFACTS_DIR / nombre / "posproceso.json")
    return modelo, config_post, device, meta


def segmentar(img_rgb, modelo, config_post, device):
    """Prepara la imagen como en el entrenamiento, predice y devuelve máscaras al tamaño original."""
    alto, ancho = img_rgb.shape[:2]
    x = normalizar(preprocesar_imagen(img_rgb))[None].to(device)
    with torch.no_grad():
        probs = torch.sigmoid(modelo(x)).cpu().numpy()[:, :, :IN_H, :IN_W]
    masks = aplicar(probs, config_post, "C")[0]
    originales = np.stack([cv2.resize(m.astype(np.uint8), (ancho, alto), interpolation=cv2.INTER_NEAREST) > 0
                           for m in masks])
    return originales, probs[0]


def mostrar(destino, imagen, caption):
    try:
        destino.image(imagen, caption=caption, width="stretch")
    except Exception:
        destino.image(imagen, caption=caption, use_container_width=True)


def chip(label):
    return (f"<span style='background:{COLORES[label]};color:white;padding:2px 10px;"
            f"border-radius:4px;margin-right:6px'>{label}</span>")


def paquete_descarga(nombre_img, masks, resumen):
    buffer = io.BytesIO()
    base = Path(nombre_img).stem
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for k, label in enumerate(LABELS):
            ok, png = cv2.imencode(".png", masks[k].astype(np.uint8) * 255)
            z.writestr(f"{base}_{label}.png", png.tobytes())
        rle = pd.DataFrame({"Image_Label": [f"{nombre_img}_{l}" for l in LABELS],
                            "EncodedPixels": [mask_to_rle(masks[k]) for k in range(len(LABELS))]})
        z.writestr(f"{base}_rle.csv", rle.to_csv(index=False))
        z.writestr(f"{base}_resumen.csv", resumen.to_csv(index=False))
    return buffer.getvalue()


st.title("Segmentación de patrones de nubes someras")
st.caption("Fish, Flower, Gravel y Sugar en imágenes satelitales MODIS. Proyecto de Visión por Computador Aplicado.")

disponibles = modelos_disponibles()
if not disponibles:
    st.error("No hay modelos con pesos en `artifacts/`. Ejecuta primero los notebooks de modelos y el del campeón, "
             "y trae los artefactos al repositorio local con `git pull`.")
    st.stop()

champion = leer_json(CHAMPION_JSON)
por_defecto = champion.get("modelo") if champion.get("modelo") in disponibles else disponibles[0]

with st.sidebar:
    st.header("Modelo")
    if champion:
        st.success(f"Campeón: **{champion['nombre']}**")
        st.caption(f"Criterio {champion['criterio']}. {champion['metrica']} en validación "
                   f"{champion['validacion']:.4f} y en prueba {champion['prueba']:.4f}.")
    else:
        st.warning("No existe `artifacts/champion.json`. Ejecuta `05_champion.ipynb` para elegir el campeón.")
    elegido = st.selectbox("Modelo a usar", disponibles, index=disponibles.index(por_defecto),
                           format_func=lambda m: MODELOS[m] + ("  (campeón)" if m == champion.get("modelo") else ""))
    resumen_modelo = leer_json(ARTIFACTS_DIR / elegido / "resumen.json")
    if resumen_modelo:
        st.caption(f"{resumen_modelo['metrica_principal']} en prueba (C): {resumen_modelo['prueba']['C']:.4f}. "
                   f"{resumen_modelo['n_params'] / 1e6:.1f} M parámetros.")
    st.header("Visualización")
    alpha = st.slider("Opacidad de las máscaras", 0.1, 0.9, 0.4, 0.05)
    ver_probs = st.checkbox("Mostrar mapas de probabilidad", value=False)

modelo, config_post, device, meta = cargar_modelo(elegido)

archivo = st.file_uploader("Sube una imagen satelital (JPG o PNG)", type=["jpg", "jpeg", "png"])
if archivo is None:
    st.info("La imagen se reduce a 350x525 como en el entrenamiento, se segmenta y las máscaras se devuelven "
            "al tamaño original. Las imágenes del dataset miden 2100x1400.")
    st.stop()

datos_img = np.frombuffer(archivo.getvalue(), dtype=np.uint8)
img = cv2.imdecode(datos_img, cv2.IMREAD_COLOR)
if img is None:
    st.error("No se pudo leer la imagen.")
    st.stop()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with st.spinner(f"Segmentando con {MODELOS[elegido]} en {device.type.upper()}..."):
    masks, probs = segmentar(img, modelo, config_post, device)

st.markdown(" ".join(chip(l) for l in LABELS), unsafe_allow_html=True)
col1, col2 = st.columns(2)
mostrar(col1, img, f"Original ({img.shape[1]}x{img.shape[0]})")
mostrar(col2, superponer(img, masks, alpha), f"Segmentación con {MODELOS[elegido]}")

area = masks.reshape(len(LABELS), -1).mean(axis=1) * 100
resumen = pd.DataFrame({"Clase": LABELS, "Presente": ["Sí" if a > 0 else "No" for a in area],
                        "% del área de la imagen": np.round(area, 1)})
st.subheader("Resultado por clase")
st.dataframe(resumen, hide_index=True)

if ver_probs:
    st.subheader("Mapas de probabilidad (350x525, antes del posprocesado)")
    columnas = st.columns(len(LABELS))
    cmap = matplotlib.colormaps["viridis"]
    for k, (col, label) in enumerate(zip(columnas, LABELS)):
        mostrar(col, (cmap(probs[k])[..., :3] * 255).astype(np.uint8), label)

st.download_button("Descargar máscaras (PNG), RLE y resumen", paquete_descarga(archivo.name, masks, resumen),
                   file_name=f"{Path(archivo.name).stem}_segmentacion.zip", mime="application/zip")
