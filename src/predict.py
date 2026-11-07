"""
Módulo de predicción para el clasificador de imágenes CIFAR-10.

Responsabilidades:
- Cargar el modelo entrenado (.keras).
- Preprocesar imágenes (redimensionar a 32x32 y normalizar a [0, 1]).
- Devolver la categoría predicha y su confianza.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_model.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"

# Clases por defecto (CIFAR-10 en español), usadas si no existe class_names.json
DEFAULT_CLASS_NAMES = [
    "Avion",
    "Automovil",
    "Ave",
    "Gato",
    "Ciervo",
    "Perro",
    "Rana",
    "Caballo",
    "Barco",
    "Camion",
]

# Dimensiones esperadas por el modelo
IMG_SIZE = (32, 32)

# ---------------------------------------------------------------------------
# Carga del modelo y clases (una sola vez al importar)
# ---------------------------------------------------------------------------
_model: tf.keras.Model | None = None
_class_names: list[str] | None = None


def _load_model() -> tf.keras.Model:
    """Carga el modelo .keras desde disco (con caché en memoria)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en: {MODEL_PATH}\n"
                "Asegúrate de haber descargado cnn_model.keras en la carpeta models/."
            )
        _model = tf.keras.models.load_model(str(MODEL_PATH))
    return _model


def _load_class_names() -> list[str]:
    """Carga los nombres de las categorías desde class_names.json."""
    global _class_names
    if _class_names is None:
        if CLASS_NAMES_PATH.exists():
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                _class_names = json.load(f)
        else:
            _class_names = list(DEFAULT_CLASS_NAMES)
    return _class_names


# ---------------------------------------------------------------------------
# Preprocesamiento de imágenes
# ---------------------------------------------------------------------------
def preparar_imagen(
    image: Union[str, Path, Image.Image, np.ndarray],
) -> np.ndarray:
    """
    Prepara una imagen para ser ingresada al modelo.

    Parámetros
    ----------
    image : str | Path | PIL.Image.Image | np.ndarray
        Ruta al archivo de imagen, objeto PIL Image o array numpy.

    Retorna
    -------
    np.ndarray
        Array de forma (1, 32, 32, 3), tipo float32, valores en [0, 1].
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        raise TypeError(
            f"Tipo de imagen no soportado: {type(image)}. "
            "Usa una ruta (str/Path), PIL.Image o np.ndarray."
        )

    # Redimensionar a lo que espera el modelo (32x32)
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

    # Convertir a array y normalizar igual que en el entrenamiento (/255)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Agregar dimensión de batch -> (1, 32, 32, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def predecir(
    image: Union[str, Path, Image.Image, np.ndarray],
) -> Tuple[str, float]:
    """
    Clasifica una imagen y devuelve la categoría y su confianza.

    Parámetros
    ----------
    image : str | Path | PIL.Image.Image | np.ndarray
        Imagen a clasificar.

    Retorna
    -------
    tuple[str, float]
        (categoria_predicha, confianza) donde confianza está entre 0 y 1.
    """
    model = _load_model()
    class_names = _load_class_names()

    x = preparar_imagen(image)

    # Predicción: shape (1, 10)
    preds = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(preds))
    categoria = class_names[idx]
    confianza = float(preds[idx])

    return categoria, confianza


# ---------------------------------------------------------------------------
# Prueba rápida directa: python src/predict.py <ruta_a_imagen>
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python src/predict.py <ruta_a_imagen>")
        sys.exit(1)

    categoria, confianza = predecir(sys.argv[1])
    print(f"Categoría: {categoria}")
    print(f"Confianza: {confianza * 100:.2f}%")
