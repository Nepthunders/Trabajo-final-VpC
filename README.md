# Segmentación de patrones de nubes someras en imágenes satelitales MODIS

Trabajo final de Visión por Computador Aplicado (Maestría en Ciencia de los Datos y Analítica). Segmentación multi-etiqueta de los patrones Fish, Flower, Gravel y Sugar con el dataset *Understanding Clouds from Satellite Images* de Kaggle.

**Por** María José Bernal, Alejandro Olivares, Santiago Zapata y Jerly Alejandra Galeano.

## Estructura

| Ruta | Contenido |
|---|---|
| `notebooks/00_eda.ipynb` | Análisis exploratorio de datos |
| `notebooks/01_cnn.ipynb` a `04_deeplab_r34.ipynb` | Un notebook por modelo, con implementación, entrenamiento, evaluación y visualización |
| `notebooks/05_champion.ipynb` | Comparación de los cuatro modelos, bootstrap y elección del campeón |
| `app/app.py` | App de Streamlit que segmenta imágenes nuevas con el campeón |
| `src/` | Código común a todos los modelos (datos, pérdida, métricas, posprocesado, gráficas) |
| `src/arquitecturas/` | Implementación de cada modelo, escrita por su notebook |
| `artifacts/<modelo>/` | Pesos, historial, métricas, posprocesado, figuras y `resultados.pkl` de cada modelo |
| `artifacts/champion.json` | Marca del modelo campeón, que es lo que lee la app |
| `artifacts/eda/` | Figuras y tablas del EDA |
| `splits/split.csv` | Partición única 80/10/10 que usan todos los modelos |
| `ejecutar_en_kaggle.ipynb` | Ejecuta todo el proyecto en Kaggle y sube los resultados |

## Protocolo común

Todos los modelos se entrenan y evalúan igual, para que la comparación sea válida. Los valores viven en `src/config.py`.

| Elemento | Valor |
|---|---|
| Tamaño | 350x525 (1/4 del original), con relleno negro hasta 352x544 |
| Partición | 80/10/10 estratificada por combinación de etiquetas, semilla 44 (4438 / 553 / 555) |
| Salida | 4 máscaras, una por clase, con sigmoide independiente (las clases se solapan) |
| Pérdida | Focal (γ = 2) + Dice |
| Aumento | Flips, transformación afín suave y cambios leves de brillo y contraste |
| Optimización | AdamW + OneCycleLR, 12 épocas, lote de 16, precisión mixta |
| Selección de la época | Mejor métrica principal en validación (umbral 0.5) |
| Posprocesado | A umbral 0.5, B umbral por clase, C umbral + área mínima + morfología, calibrados en validación |
| Evaluación | Una sola vez en prueba, sobre 350x525 sin el relleno |
| Métrica principal | Dice global (`METRICA_PRINCIPAL = "dice"`); con `"dice_kaggle"` se usa el Dice de la competencia |
| Métricas complementarias | IoU, Dice de la competencia, precisión y recall por píxel y por imagen |

## Modelos

| Notebook | Modelo | Encoder | Parámetros | Qué pone a prueba |
|---|---|---|---|---|
| `01_cnn` | CNN Segmenter | Propio, desde cero | 7.0 M | Línea base, sin puentes ni preentrenamiento |
| `02_att_resunet` | Attention Residual U-Net | Propio, desde cero | 8.2 M | Puentes con atención y bloques residuales |
| `03_unet_r34` | U-Net | ResNet34, ImageNet | 24.4 M | Puentes más transferencia de aprendizaje |
| `04_deeplab_r34` | DeepLabV3+ | ResNet34, ImageNet | 22.4 M | Contexto multiescala (ASPP) frente a puentes, con el mismo encoder |

## Cómo ejecutar en Kaggle

1. Crear un notebook desde la pestaña **Code** de la competencia, con Internet activado, GPU T4 x2 y el Secret `GITHUB_TOKEN`.
2. Copiar ahí el contenido de `ejecuccion_kaggle.ipynb` y cambiar `REPO` y el correo.
3. Ejecutar las celdas en orden. Al final, todos los notebooks quedan ejecutados con sus resultados y se suben al repositorio junto con `artifacts/`.

Para repetir la evaluación sin volver a entrenar, poner `ENTRENAR = False` en el notebook del modelo. Así usa los pesos guardados en `artifacts/<modelo>/`.

## Cambiar el campeón

`05_champion.ipynb` elige automáticamente el modelo con mayor métrica principal en validación con el posprocesado completo. Para forzar otro, hay dos formas.
- Poner su nombre en `CHAMPION_MANUAL` (por ejemplo `"unet_r34"`) y volver a ejecutar el notebook.
- Editar a mano el campo `modelo` de `artifacts/champion.json`.

La app también permite escoger cualquier modelo con pesos desde el panel lateral.

## Cómo usar la app en un computador local

Desde la raíz del repositorio, con los artefactos ya descargados (`git pull`):

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

La app se abre en el navegador. Recibe una imagen JPG o PNG, la prepara igual que en el entrenamiento, aplica el modelo campeón con su posprocesado y muestra las máscaras superpuestas, el porcentaje de área de cada clase y un botón para descargar las máscaras en PNG y en RLE. Funciona en CPU.
