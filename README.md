# Predicción Espacio-Temporal de Delitos en Lima Metropolitana

---

## Descripción General

Este proyecto implementa un **modelo integrado de grafos** para predecir la **tasa de delitos por distrito** en **Lima Metropolitana**, usando datos anuales de denuncias policiales del 2016 al 2023.

El modelo está **inspirado y adaptado** del paper:
> **"An Integrated Graph Model for Spatial–Temporal Urban Crime Prediction Based on Attention Mechanism"**

### Arquitectura del Modelo

El sistema integra múltiples componentes de deep learning:

1. **Bloque MLP Residual (ResNet-like)**: Procesa características iniciales por nodo y paso temporal
2. **Red de Convolución de Grafos (GCN)**: Captura relaciones espaciales entre distritos mediante una matriz de adyacencia basada en proximidad geográfica
3. **LSTM Bidireccional**: Modela dependencias temporales en las series de tiempo
4. **Mecanismo de Atención Temporal**: Pondera la importancia de diferentes pasos temporales
5. **Proyección Final**: Genera predicciones por distrito

El objetivo es explorar patrones espacio-temporales de criminalidad y generar **predicciones futuras** (por ejemplo, para el año 2024).

---

## Requisitos

### Pre-requisitos

1. **Python 3.8 o superior**
2. **CUDA** (opcional, para aceleración con GPU)

### Dependencias

Las dependencias principales incluyen:
- `pandas`, `numpy` para procesamiento de datos
- `scikit-learn` para normalización y métricas
- `torch` (PyTorch) para el modelo de deep learning

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/bladimirAlfer/Proyecto_CDD.git
cd Proyecto_CDD
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

**Nota**: Si tienes GPU disponible y quieres usar CUDA, instala PyTorch con soporte CUDA desde [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Estructura de Datos

### Formato de los Archivos CSV

El pipeline espera archivos CSV con el siguiente formato:
- `final2016.csv`, `final2017.csv`, ..., `final2023.csv` en la carpeta `data/`

Cada CSV debe contener las siguientes columnas:
- `anio`: Año del incidente (numérico)
- `X`: Coordenada X (longitud)
- `Y`: Coordenada Y (latitud)
- `distrito`: Nombre del distrito (texto)

### Preparación de Datos

Asegúrate de que los archivos `final*.csv` estén en la carpeta `data/`:

```
Proyecto_CDD/
├── data/
│   ├── final2016.csv
│   ├── final2017.csv
│   ├── ...
│   └── final2023.csv
├── main_version_preliminar
├── requirements.txt
└── README.md
```

---

## Instrucciones Paso a Paso

### Paso 1: Verificar los Datos

Asegúrate de que todos los archivos CSV estén en la carpeta `data/` y tengan el formato correcto:

```bash
# Verificar que los archivos existen
ls data/final*.csv
```

### Paso 2: Configurar el Pipeline (Opcional)

Si necesitas ajustar parámetros, edita las variables en `main_version_preliminar`:

```python
@dataclass
class CFG:
    data_glob: str = "./data/final*.csv"  # Patrón de archivos
    seq_len: int = 3                       # Longitud de ventana temporal
    batch_size: int = 64                   # Tamaño de batch
    epochs: int = 40                       # Número de épocas
    lr: float = 1e-3                       # Learning rate
    hidden_dim: int = 64                   # Dimensión oculta
    gcn_k: int = 4                         # Número de vecinos para GCN
```

### Paso 3: Ejecutar el Pipeline

Ejecuta el script principal:

```bash
python main_version_preliminar
```

### Paso 4: Proceso Automático

El pipeline realizará automáticamente:

1. **Carga de datos**: Lee y concatena todos los CSV que coinciden con el patrón
2. **Normalización**: Estandariza tipos y formatos, calcula centroides por distrito
3. **Construcción del grafo**: Crea matriz de adyacencia basada en k-vecinos más cercanos (kNN)
4. **Agregación temporal**: Convierte datos a serie temporal (años × distritos)
5. **División temporal**: Split 70% entrenamiento / 30% prueba
6. **Escalado**: Normalización usando solo datos de entrenamiento
7. **Entrenamiento**: Entrena el modelo integrado con early stopping basado en RMSE de validación
8. **Evaluación**: Calcula métricas (MAE, RMSE, MAPE) en conjunto de prueba
9. **Guardado**: Guarda el mejor modelo y genera reporte JSON

### Paso 5: Revisar Resultados

Los resultados se guardan en la carpeta `ig_outputs/`:

- `integrated_model.pt`: Modelo entrenado con mejor rendimiento en validación
- `report.json`: Reporte con métricas de evaluación

```bash
# Ver el reporte
cat ig_outputs/report.json
```

---

## Métricas de Evaluación

El modelo reporta las siguientes métricas:

- **MAE (Mean Absolute Error)**: Error absoluto medio
- **RMSE (Root Mean Squared Error)**: Raíz del error cuadrático medio
- **MAPE (Mean Absolute Percentage Error)**: Error porcentual absoluto medio (solo para valores no-cero)

---

## Configuración Avanzada

### Cambiar el Dispositivo (CPU/GPU)

El script detecta automáticamente si hay GPU disponible. Para forzar CPU:

```python
cfg.device = "cpu"
```

### Ajustar la Ventana Temporal

Para cambiar la longitud de la ventana temporal (por defecto 3 años):

```python
cfg.seq_len = 5  # Usa 5 años de historia
```

### Modificar el Número de Vecinos en el Grafo

Para cambiar cuántos vecinos conecta cada distrito:

```python
cfg.gcn_k = 6  # Conecta con 6 vecinos más cercanos
```

---

## Notas Técnicas

- **Matriz de Adyacencia**: Se construye usando distancia euclidiana entre centroides de distritos. Si tienes información de límites administrativos reales, puedes reemplazar esta lógica.
- **Escalado**: Se usa `StandardScaler` de scikit-learn, ajustado solo con datos de entrenamiento para evitar data leakage.
- **Validación**: Se usa una ventana deslizante de los últimos `2*seq_len` pasos del conjunto de entrenamiento.
- **Guardado del Modelo**: El modelo guardado incluye el estado del modelo, el scaler, nombres de distritos, matriz de adyacencia y configuración.

---

## Referencias

Este proyecto está inspirado en:
- **Paper**: "An Integrated Graph Model for Spatial–Temporal Urban Crime Prediction Based on Attention Mechanism"

---

## Estructura del Proyecto

```
Proyecto_CDD/
├── data/                    # Datos CSV (final2016.csv, ..., final2023.csv)
├── ig_outputs/              # Salidas del modelo (generado automáticamente)
│   ├── integrated_model.pt # Modelo entrenado
│   └── report.json         # Reporte de métricas
├── main_version_preliminar  # Script principal del pipeline
├── requirements.txt         # Dependencias Python
└── README.md               # Este archivo
```

---

## Solución de Problemas

### Error: "No se encontraron CSV con el patrón"

Verifica que los archivos estén en `data/` y sigan el patrón `final*.csv`.

### Error: "La serie tiene solo X pasos"

Reduce `seq_len` o asegúrate de tener suficientes años de datos.

### Error: "Falta columna requerida"

Verifica que tus CSV tengan las columnas: `anio`, `X`, `Y`, `distrito`.

---

## Licencia

[Especificar licencia si aplica]
