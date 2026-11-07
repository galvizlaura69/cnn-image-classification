"""
Interfaz de Streamlit para el clasificador de imágenes CIFAR-10.

Ejecuta con:
    streamlit run app/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Asegurar que se pueda importar src.predict sin importar el directorio de trabajo
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
from PIL import Image

from src.predict import (
    _load_class_names,
    _load_model,
    predecir,
    preparar_imagen,
)

st.set_page_config(
    page_title="Clasificador de Imágenes CIFAR-10",
    page_icon="🖼️",
    layout="centered",
)

st.title("🖼️ Clasificador de Imágenes CIFAR-10")
st.write(
    "Sube una imagen y el modelo la clasificará en una de las 10 categorías de CIFAR-10. "
    "El modelo fue entrenado en Google Colab y se carga desde `models/cnn_model.keras`."
)

st.divider()

archivo = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"],
    help="Formatos soportados: JPG, JPEG, PNG",
)

if archivo is not None:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen subida", use_container_width=True)

    if st.button("Clasificar imagen", type="primary"):
        with st.spinner("Clasificando..."):
            categoria, confianza = predecir(imagen)

        st.success("¡Clasificación completada!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Categoría predicha", categoria)
        with col2:
            st.metric("Confianza", f"{confianza * 100:.2f}%")

        st.progress(confianza)

        # Barra de probabilidades por categoría
        st.subheader("Probabilidades por categoría")
        model = _load_model()
        class_names = _load_class_names()
        x = preparar_imagen(imagen)
        preds = model.predict(x, verbose=0)[0]

        orden = np.argsort(preds)[::-1]
        datos = {
            "Categoría": [class_names[i] for i in orden],
            "Probabilidad": [float(preds[i]) for i in orden],
        }
        st.bar_chart(datos, x="Categoría", y="Probabilidad", height=350)

else:
    st.info("👆 Sube una imagen para comenzar.")

st.divider()
st.caption(
    "Clases del modelo: Avion, Automovil, Ave, Gato, Ciervo, Perro, Rana, Caballo, Barco, Camion"
)
