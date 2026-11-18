"""Predicción, evaluación por variantes de posprocesado y guardado de los artefactos de cada modelo."""
import copy
import json
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import H, LABELS, METRICA_PRINCIPAL, MODELOS, PROTOCOLO, W, carpeta_modelo
from datos import CloudDataset
from metricas import conteos, promedio, recortar, tabla
from posproceso import VARIANTES, aplicar


@torch.no_grad()
def predecir(modelo, split, device, lote=None):
    """Probabilidades (N, 4, 350, 525) en float16, etiquetas booleanas y milisegundos por imagen."""
    lote = lote or PROTOCOLO["lote"] * 2
    dl = DataLoader(CloudDataset(split), batch_size=lote, shuffle=False, num_workers=PROTOCOLO["workers"])
    modelo.eval()
    probs, targets, t_total = [], [], 0.0
    for x, y in dl:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = modelo(x.to(device))
        p = torch.sigmoid(logits.float())
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total += time.perf_counter() - t0
        probs.append(recortar(p.cpu().numpy()).astype(np.float16))
        targets.append(recortar(y.numpy() > 0.5))
    probs, targets = np.concatenate(probs), np.concatenate(targets)
    return probs, targets, 1000 * t_total / len(probs)


@torch.no_grad()
def tiempo_cpu_ms(modelo, repeticiones=3):
    """Milisegundos por imagen en CPU, que es como correrá la app en un computador sin GPU."""
    m = copy.deepcopy(modelo).float().cpu().eval()
    x = torch.randn(1, 3, H, W)
    m(x)
    t0 = time.perf_counter()
    for _ in range(repeticiones):
        m(x)
    return 1000 * (time.perf_counter() - t0) / repeticiones


def evaluar_variantes(probs, targets, config):
    """Conteos y tabla resumen de las tres variantes de posprocesado."""
    cnt = {v: conteos(aplicar(probs, config, v), targets) for v in VARIANTES}
    filas = []
    for v, c in cnt.items():
        t = tabla(c)
        fila = {"variante": VARIANTES[v]}
        fila.update(t["Promedio"].to_dict())
        fila.update({f"dice_{l}": t.loc["dice", l] for l in LABELS})
        filas.append(fila)
    return cnt, pd.DataFrame(filas).set_index("variante")


def guardar_resultados(nombre, historial, config_post, cnt_val, cnt_test, tabla_val, tabla_test, info):
    """Escribe en artifacts/<modelo>/ todo lo necesario para comparar y desplegar sin volver a entrenar."""
    carpeta = carpeta_modelo(nombre)
    historial.to_csv(carpeta / "historial.csv", index=False)
    with open(carpeta / "posproceso.json", "w") as f:
        json.dump(config_post, f, indent=2)

    largas = []
    for split, cnt in (("val", cnt_val), ("test", cnt_test)):
        for v, c in cnt.items():
            t = tabla(c).reset_index(names="metrica")
            t.insert(0, "variante", v)
            t.insert(0, "split", split)
            largas.append(t)
    metricas = pd.concat(largas, ignore_index=True)
    metricas.to_csv(carpeta / "metricas.csv", index=False)

    np.savez_compressed(carpeta / "conteos.npz", **{f"{s}_{v}": c for s, cnt in (("val", cnt_val), ("test", cnt_test))
                                                    for v, c in cnt.items()})

    resumen = {
        "modelo": nombre,
        "nombre": MODELOS[nombre],
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "metrica_principal": METRICA_PRINCIPAL,
        "validacion": {v: round(promedio(c, METRICA_PRINCIPAL), 4) for v, c in cnt_val.items()},
        "prueba": {v: round(promedio(c, METRICA_PRINCIPAL), 4) for v, c in cnt_test.items()},
        **info,
    }
    with open(carpeta / "resumen.json", "w") as f:
        json.dump(resumen, f, indent=2)

    with open(carpeta / "resultados.pkl", "wb") as f:
        pickle.dump({"resumen": resumen, "protocolo": PROTOCOLO, "historial": historial, "posproceso": config_post,
                     "metricas": metricas, "tabla_validacion": tabla_val, "tabla_prueba": tabla_test,
                     "conteos_validacion": cnt_val, "conteos_prueba": cnt_test}, f)

    print(f"Artefactos guardados en {carpeta}:")
    for archivo in sorted(carpeta.iterdir()):
        print(f"  {archivo.name:22s} {archivo.stat().st_size / 1e6:8.2f} MB")
    return resumen
