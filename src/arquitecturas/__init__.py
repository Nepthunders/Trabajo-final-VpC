"""Cada archivo de esta carpeta lo escribe su notebook con %%writefile y expone construir_modelo()."""
import importlib


def construir(nombre, preentrenado=True):
    modulo = importlib.import_module(f"arquitecturas.{nombre}")
    return modulo.construir_modelo(preentrenado=preentrenado)
