# Clasificador de Imágenes CIFAR-10 con CNN

Proyecto de Deep Learning que utiliza una Red Neuronal Convolucional (CNN) para clasificar imágenes en 10 categorías del dataset CIFAR-10. Incluye notebook de experimentación en Google Colab, código de predicción, modelo entrenado y aplicación web con Streamlit.

## 1. Dataset

Se utilizó el **dataset CIFAR-10**, que contiene 60,000 imágenes a color de 32×32 píxeles divididas en 10 clases:

| # | Categoría | Imágenes |
|---|-----------|----------|
| 1 | Avión | 6,000 |
| 2 | Automóvil | 6,000 |
| 3 | Ave | 6,000 |
| 4 | Gato | 6,000 |
| 5 | Ciervo | 6,000 |
| 6 | Perro | 6,000 |
| 7 | Rana | 6,000 |
| 8 | Caballo | 6,000 |
| 9 | Barco | 6,000 |
| 10 | Camión | 6,000 |

## 2. Estructura del repositorio

```
cnn-image-classification/
├── .gitignore
├── README.md
├── requirements.txt
├── notebooks/
│   └── cnn_experiments.ipynb      # Notebook de experimentación (Google Colab)
├── src/
│   ├── __init__.py
│   └── predict.py                 # Código de predicción (cargar modelo, preprocesar, predecir)
├── app/
│   └── app.py                     # Aplicación Streamlit
├── models/
│   ├── cnn_model.keras            # Modelo entrenado guardado
│   ├── class_names.json           # Nombres de las 10 categorías
│   └── metrics.json               # Métricas de evaluación
└── results/
    └── predictions/               # Predicciones de ejemplo
```

## 3. Requisitos

- Python 3.11+
- TensorFlow 2.21.0
- Streamlit 1.67.0
- Pillow, NumPy

Todas las dependencias están listadas en `requirements.txt`.

## 4. Instalación

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/galvizlaura69/cnn-image-classification.git
cd cnn-image-classification
```

### Paso 2: Crear y activar el entorno virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Paso 3: Instalar dependencias

```powershell
python -m pip install -r requirements.txt
```

> Nota: Si `pip install tensorflow` falla por rutas largas en Windows, habilita `LongPathsEnabled` en el registro o mueve el entorno a una carpeta corta (ej. `C:\venv`).

## 5. Ejecutar la aplicación Streamlit

Desde la carpeta del proyecto, con el entorno virtual activado:

```powershell
streamlit run app/app.py
```

La aplicación se abrirá en tu navegador (por defecto en `http://localhost:8501`).

### Funcionalidad de la app

1. Sube una imagen (JPG, JPEG o PNG).
2. Haz clic en **"Clasificar imagen"**.
3. La app mostrará:
   - La **categoría predicha** (entre las 10 clases de CIFAR-10).
   - El **nivel de confianza** (probabilidad).
   - Un gráfico de barras con la probabilidad de cada categoría.

## 6. Usar el módulo de predicción (src/predict.py)

Puedes importar y usar `predecir()` directamente en tu propio código:

```python
from src.predict import predecir

categoria, confianza = predecir("ruta/a/tu/imagen.jpg")
print(f"Categoría: {categoria}")
print(f"Confianza: {confianza * 100:.2f}%")
```

También puedes ejecutarlo como script desde la terminal:

```powershell
python src/predict.py ruta/a/tu/imagen.jpg
```

## 7. Arquitectura del modelo

La CNN tiene las siguientes capas:

1. `Conv2D(32, 3×3)` + `MaxPooling2D`
2. `Conv2D(64, 3×3)` + `MaxPooling2D`
3. `Conv2D(128, 3×3)` + `MaxPooling2D`
4. `Flatten`
5. `Dense(128, activation="relu")` + `Dropout(0.5)`
6. `Dense(10, activation="softmax")`

- Entrada: imágenes de 32×32×3, normalizadas a [0, 1]
- Salida: probabilidades para las 10 categorías
- Total de parámetros: 356,810

## 8. Resultados

- **Precisión en prueba:** 74.82%
- **Pérdida en prueba:** 0.9004

Se incluyen en el notebook:
- Gráficas de precisión y pérdida por época
- Matriz de confusión
- Clasificación por categoría (precision, recall, f1-score)
- 10 predicciones de ejemplo con imágenes

## 9. Notebook de experimentación

El archivo `notebooks/cnn_experiments.ipynb` contiene todo el desarrollo realizado en Google Colab:
- Carga y exploración del dataset CIFAR-10
- Visualización de imágenes
- Normalización de píxeles
- División entrenamiento/validación/prueba
- Construcción y entrenamiento de la CNN (2 épocas principales)
- Data Augmentation (volteo, rotación, zoom)
- Evaluación con métricas y matriz de confusión
- Guardado del modelo en formato `.keras`

## 10. Modelo entrenado

El modelo ya entrenado se encuentra en `models/cnn_model.keras` (formato Keras nativo). No es necesario volver a entrenar; la aplicación Streamlit carga directamente este archivo.

## 11. Autores

Proyecto desarrollado para la materia de Machine Learning / Deep Learning.

---

### Licencia

Este proyecto es de carácter educativo.