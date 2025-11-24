# Sistema de Predicción Espacio-Temporal de Delitos en Lima Metropolitana

---

## Aplicación en Línea

**La aplicación web está desplegada y disponible en:**

🔗 **[https://bladimiralfer-proyecto-cdd-app-ii1kxc.streamlit.app/](https://bladimiralfer-proyecto-cdd-app-ii1kxc.streamlit.app/)**

La aplicación permite visualizar predicciones de delitos, explorar datos históricos, generar reportes y analizar tendencias por distrito mediante una interfaz interactiva con mapas 3D y visualizaciones avanzadas.

---

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Componentes del Modelo](#componentes-del-modelo)
4. [Decisiones de Diseño](#decisiones-de-diseño)
5. [Instalación y Configuración](#instalación-y-configuración)
6. [Uso del Sistema](#uso-del-sistema)
7. [Aplicación Web (Streamlit)](#aplicación-web-streamlit)
8. [API y Funcionalidades](#api-y-funcionalidades)
9. [Deployment](#deployment)
10. [Métricas y Evaluación](#métricas-y-evaluación)
11. [Estructura del Proyecto](#estructura-del-proyecto)
12. [Referencias](#referencias)
13. [Solución de Problemas](#solución-de-problemas)

---

## Descripción General

Este proyecto implementa un **sistema integrado de predicción espacio-temporal de delitos** para Lima Metropolitana, utilizando datos anuales de denuncias policiales desde 2016 hasta 2023. El sistema proporciona predicciones a nivel distrital mediante dos enfoques complementarios:

1. **Modelo Integrado de Grafos (GCN + LSTM + Attention)**: Arquitectura de deep learning que combina redes convolucionales de grafos, redes neuronales recurrentes y mecanismos de atención para capturar patrones espacio-temporales complejos.

2. **Modelo Baseline (Random Forest / XGBoost)**: Enfoque basado en árboles de decisión con features temporales y espaciales para comparación y validación.

### Objetivos

- Predecir tasas de delitos por distrito para años futuros (ej: 2024)
- Identificar patrones espacio-temporales en la criminalidad urbana
- Proporcionar una interfaz interactiva para visualización y análisis
- Facilitar la toma de decisiones para asignación de recursos policiales

### Inspiración Académica

Este proyecto está inspirado y adaptado del siguiente trabajo de investigación:

> **"An Integrated Graph Model for Spatial–Temporal Urban Crime Prediction Based on Attention Mechanism"** (Hou et al., 2022)

El modelo integrado implementa una arquitectura similar, adaptada a las características específicas de los datos de Lima Metropolitana.

---

## Arquitectura del Sistema

### Diagrama de Arquitectura General

```
┌─────────────────────────────────────────────────────────────────┐
│                     CAPA DE DATOS                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ final2016.csv│  │ final2017.csv│  │  final20XX   │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                  │                  │                 │
│         └──────────────────┴──────────────────┘                 │
│                            │                                    │
│                            ▼                                    │
│              ┌─────────────────────────┐                        │
│              │  Pipeline de Procesamiento                       │
│              │  - Normalización de esquema                      │
│              │  - Cálculo de centroides                         │
│              │  - Construcción de grafo (kNN)                   │
│              │  - Agregación temporal (año × distrito)          │
│              └─────────────┬───────────┘                        │
└────────────────────────────┼────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
┌───────────────────────────┐  ┌────────────────────────────┐
│  Modelo Integrado         │  │  Modelo Baseline           │
│  (GCN + LSTM + Attention) │  │  (RF / XGBoost)            │
│                           │  │                            │
│  ┌────────────────────┐   │  │  ┌──────────────────────┐  │
│  │ ResBlock MLP       │   │  │  │ Feature Engineering  │  │
│  │   (Embedding)      │   │  │  │ - Lags temporales    │  │
│  └────────┬───────────┘   │  │  │ - Rolling stats      │  │
│           │               │  │  │ - Shares categóricos│   │
│           ▼               │  │  └──────────┬───────────┘  │
│  ┌────────────────────┐   │  │             │              │
│  │ GCN Layer          │   │  │             ▼              │
│  │   (Relaciones      │   │  │  ┌──────────────────────┐  │
│  │    espaciales)     │   │  │  │ RF/XGB Regressor     │  │
│  └────────┬───────────┘   │  │  └──────────────────────┘  │
│           │               │  │                            │
│           ▼               │  │                            │
│  ┌────────────────────┐   │  │                            │
│  │ BiLSTM             │   │  │                            │
│  │   (Dependencias    │   │  │                            │
│  │    temporales)     │   │  │                            │
│  └────────┬───────────┘   │  │                            │
│           │               │  │                            │
│           ▼               │  │                            │
│  ┌────────────────────┐   │  │                            │
│  │ Temporal Attention │   │  │                            │
│  │   (Ponderación)    │   │  │                            │
│  └────────┬───────────┘   │  │                            │
│           │               │  │                            │
│           ▼               │  │                            │
│  ┌────────────────────┐   │  │                            │
│  │ Output Projection  │   │  │                            │
│  └────────┬───────────┘   │  │                            │
└───────────┼───────────────┴──┼────────────────────────────┘
            │                  │
            └────────┬─────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Predicciones         │
        │   (Tasa por distrito)  │
        └────────────┬───────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌─────────────────────┐
│  Aplicación      │    │  Archivos de Salida │
│  Streamlit       │    │  - integrated_model │
│  (Visualización) │    │  - report.json      │
│                  │    │  - metrics.json     │
│  - Mapa 3D       │    │  - predictions.csv  │
│  - KPIs          │    │                     │
│  - Gráficos      │    │                     │
└──────────────────┘    └─────────────────────┘
```

### Flujo de Datos del Modelo Integrado

```
Entrada: Serie temporal por distrito
┌────────────────────────────────────────┐
│ T-2  │  T-1  │  T    │  (seq_len=3)    │
│ [N]  │  [N]  │  [N]  │                 │
└──────┴───────┴───────┘                 │
         │                               │
         ▼                               │
┌─────────────────────┐                  │
│ ResBlock MLP        │                  │
│ (B, S, N, 1) →      │                  │
│ (B, S, N, hidden)   │                  │
└─────────┬───────────┘                  │
          │                              │
          ▼                              │
┌─────────────────────┐                  │
│ GCN Layer           │                  │
│ Propaga información │                  │
│ entre distritos     │                  │
│ vecinos (kNN)       │                  │
└─────────┬───────────┘                  │
          │                              │
          ▼                              │
┌─────────────────────┐                  │
│ BiLSTM              │                  │
│ Modela dependencias │                  │
│ temporales          │                  │
│ (B, N, S, F)        │                  │
└─────────┬───────────┘                  │
          │                              │
          ▼                              │
┌─────────────────────┐                  │
│ Temporal Attention  │                  │
│ Ponderación de      │                  │
│ pasos temporales    │                  │
│ (B, N, F)           │                  │
└─────────┬───────────┘                  │
          │                              │
          ▼                              │
┌─────────────────────┐                  │
│ Output Projection   │                  │
│ (B, N, F) → (B, N)  │                  │
└─────────┬───────────┘                  │
          │                              │
          ▼                              │
   Predicción T+1
      [N] (distritos)
```

---

## Componentes del Modelo

### 1. Bloque MLP Residual (ResBlockMLP)

**Propósito**: Transforma los valores de entrada (conteos de delitos) en representaciones latentes de mayor dimensionalidad, permitiendo al modelo capturar relaciones no lineales.

**Arquitectura**:
- Dos capas lineales con activación ReLU
- Conexión residual para facilitar el entrenamiento profundo
- Proyección de dimensionalidad si `in_ch != hidden`

**Dimensiones**: `(B, S, N, 1) → (B, S, N, hidden_dim)`

**Justificación**: Las conexiones residuales mejoran el flujo de gradientes y permiten entrenar arquitecturas más profundas. Este bloque procesa cada distrito y cada paso temporal de forma independiente antes de la propagación espacial.

### 2. Red de Convolución de Grafos (SimpleGCNLayer)

**Propósito**: Captura relaciones espaciales entre distritos mediante la propagación de información a través de la matriz de adyacencia del grafo.

**Construcción del Grafo**:
- **Matriz de Adyacencia**: Basada en k-vecinos más cercanos (kNN) usando distancia euclidiana entre centroides de distritos
- **Normalización**: Por grado (row-normalized) para estabilidad numérica
- **Simetría**: La matriz es simétrica (grafos no dirigidos)

**Dimensiones**: `(B, S, N, hidden) → (B, S, N, hidden)`

**Trade-offs**:
- **Ventaja**: Captura influencia espacial entre distritos vecinos sin necesidad de límites administrativos exactos
- **Limitación**: Depende de la calidad de los centroides calculados. Distritos con formas irregulares pueden tener centroides no representativos.

**Alternativas consideradas**:
- Usar límites administrativos reales (requiere datos GIS adicionales)
- Grafos dirigidos basados en flujos poblacionales (complejidad adicional)
- Multiple scales (multi-scale GCN) para capturar relaciones a diferentes distancias

### 3. LSTM Bidireccional

**Propósito**: Modela dependencias temporales secuenciales, capturando patrones de corto y largo plazo en las series de tiempo.

**Configuración**:
- Bidireccional para capturar contexto hacia adelante y atrás
- Dimensión oculta: `hidden_dim // 2` en cada dirección
- `batch_first=True` para compatibilidad con PyTorch

**Dimensiones**: `(B*N, S, hidden) → (B*N, S, hidden)`

**Justificación**: La bidireccionalidad permite al modelo considerar tanto el pasado como el "futuro contextual" de cada paso temporal, mejorando la comprensión de tendencias. Se procesa cada distrito de forma independiente en la dimensión temporal.

### 4. Mecanismo de Atención Temporal (TemporalAttention)

**Propósito**: Pondera dinámicamente la importancia de diferentes pasos temporales, permitiendo al modelo enfocarse en períodos más relevantes para la predicción.

**Implementación**:
- Atención basada en Query-Key-Value (similar a Transformers)
- Softmax sobre la dimensión temporal
- Agregación mediante suma ponderada

**Dimensiones**: `(B, S, N, hidden) → (B, N, hidden)`

**Ventajas**:
- Interpretabilidad: Los pesos de atención pueden indicar qué períodos históricos son más relevantes
- Adaptabilidad: Se ajusta automáticamente según los patrones de cada distrito

**Trade-offs**:
- **Ventaja**: Mayor flexibilidad que promedios simples o ventanas fijas
- **Costo**: Requiere parámetros adicionales y cómputo extra

### 5. Proyección Final (Output Projection)

**Propósito**: Mapea las representaciones latentes a predicciones numéricas (tasas de delitos).

**Arquitectura**:
- Dos capas lineales con activación ReLU intermedia
- Reducción gradual: `hidden_dim → hidden_dim//2 → 1`
- Salida escalar por distrito

**Dimensiones**: `(B, N, hidden) → (B, N, 1) → (B, N)`

---

## Decisiones de Diseño

### División Temporal de Datos

**Decisión**: Split temporal 70% entrenamiento / 30% prueba (no aleatorio por año)

**Justificación**:
- Preserva la naturaleza temporal de los datos
- Evita data leakage (información futura no disponible en entrenamiento)
- Simula escenario real: predecir años futuros basado en historia pasada

**Trade-offs**:
- **Ventaja**: Más realista para producción
- **Limitación**: Menos datos de prueba (solo últimos años)
- **Alternativa rechazada**: K-fold temporal (menos intuitivo para evaluación de años específicos)

### Construcción de la Matriz de Adyacencia

**Decisión**: k-Nearest Neighbors basado en distancia euclidiana entre centroides

**Justificación**:
- Simple y eficiente de calcular
- No requiere datos GIS adicionales (límites administrativos)
- Captura proximidad geográfica como proxy de influencia espacial

**Trade-offs**:
- **Ventaja**: Funciona con datos mínimos (solo coordenadas)
- **Limitación**: Puede conectar distritos no adyacentes administrativamente
- **Alternativa considerada**: Usar límites administrativos reales (requiere datos adicionales, implementación futura)

**Valor de k**: Por defecto `gcn_k=4`, configurable según análisis de conectividad espacial.

### Normalización de Datos

**Decisión**: StandardScaler ajustado solo con datos de entrenamiento

**Justificación**:
- Previene data leakage (scaler no "ve" datos de prueba)
- Establece rangos consistentes para el modelo
- Facilita convergencia del entrenamiento

**Trade-offs**:
- **Ventaja**: Práctica estándar en ML, previene overfitting
- **Consideración**: Requiere guardar parámetros del scaler para inferencia

### Ventana Temporal (seq_len)

**Decisión**: Por defecto `seq_len=3` (usa 3 años de historia para predecir el siguiente)

**Justificación**:
- Balance entre contexto histórico y capacidad de generalización
- Considera que con 8 años de datos (2016-2023), usar más de 3 reduce significativamente ejemplos de entrenamiento

**Trade-offs**:
- **Ventaja**: Suficiente contexto para capturar tendencias
- **Limitación**: Puede perder patrones de muy largo plazo
- **Alternativa**: `seq_len=5` (probablemente mejor si hay más datos históricos)

### Selección del Dispositivo (CPU vs GPU)

**Decisión**: Detección automática con fallback a CPU

**Justificación**:
- Accesibilidad: Funciona en máquinas sin GPU
- Rendimiento: Aprovecha GPU si está disponible
- Flexibilidad: Permite forzar CPU para debugging

**Trade-offs**:
- **GPU**: Entrenamiento 5-10x más rápido, pero requiere hardware adicional
- **CPU**: Más lento pero universalmente disponible

### Early Stopping

**Decisión**: Validación basada en RMSE con paciencia configurable

**Justificación**:
- Previene overfitting
- Ahorra tiempo de cómputo
- Encuentra el mejor modelo sin necesidad de entrenar todas las épocas

**Trade-offs**:
- **Ventaja**: Automatiza el proceso de selección de modelo
- **Consideración**: Puede detenerse temprano si la métrica fluctúa (requiere ajuste de paciencia)

---

## Instalación y Configuración

### Requisitos Previos

- **Python**: 3.8 o superior
- **CUDA** (opcional): Para aceleración con GPU
- **Memoria**: Mínimo 8GB RAM recomendado
- **Espacio en disco**: ~500MB para datos y modelos

### Instalación Paso a Paso

#### 1. Clonar el Repositorio

```bash
git clone https://github.com/bladimirAlfer/Proyecto_CDD.git
cd Proyecto_CDD
```

#### 2. Crear Entorno Virtual (Recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

#### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Nota sobre PyTorch**: Si tienes GPU disponible y quieres usar CUDA, instala PyTorch con soporte CUDA desde [pytorch.org](https://pytorch.org/get-started/locally/) antes de instalar las otras dependencias. Por ejemplo:

```bash
# Ejemplo para CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### 4. Verificar Instalación

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

### Estructura de Datos Requerida

El pipeline espera archivos CSV en la carpeta `data/` con el siguiente formato:

- Archivos: `final2016.csv`, `final2017.csv`, ..., `final2023.csv`
- Columnas requeridas:
  - `anio`: Año del incidente (numérico, ej: 2016)
  - `X`: Coordenada X / Longitud (numérico)
  - `Y`: Coordenada Y / Latitud (numérico)
  - `distrito`: Nombre del distrito (texto)

**Ejemplo de datos**:

```csv
anio,X,Y,distrito
2016,-77.042793,-12.046374,LIMA
2016,-77.028240,-12.087502,MIRAFLORES
...
```

### Configuración del Pipeline

Los parámetros principales se definen en `main_version_preliminar.py` mediante la clase `CFG`:

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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    outdir: str = "ig_outputs"             # Directorio de salida
```

---

## Uso del Sistema

### Entrenamiento del Modelo Integrado

#### Ejecución Básica

```bash
python main_version_preliminar.py
```

Este comando ejecuta el pipeline completo:

1. **Carga de datos**: Lee y concatena todos los CSV que coinciden con `data/final*.csv`
2. **Normalización**: Estandariza tipos y formatos, calcula centroides por distrito
3. **Construcción del grafo**: Crea matriz de adyacencia basada en k-vecinos más cercanos (kNN)
4. **Agregación temporal**: Convierte datos a serie temporal (años × distritos)
5. **División temporal**: Split 70% entrenamiento / 30% prueba
6. **Escalado**: Normalización usando solo datos de entrenamiento
7. **Entrenamiento**: Entrena el modelo integrado con early stopping basado en RMSE de validación
8. **Evaluación**: Calcula métricas (MAE, RMSE, MAPE) en conjunto de prueba
9. **Guardado**: Guarda el mejor modelo y genera reporte JSON

#### Salidas Generadas

El pipeline genera los siguientes archivos en `ig_outputs/`:

- `integrated_model.pt`: Modelo entrenado con mejor rendimiento en validación
  - Contiene: estado del modelo, scaler (mean, scale), nombres de distritos, matriz de adyacencia, configuración
- `report.json`: Reporte con métricas de evaluación (MAE, RMSE, MAPE)

**Ejemplo de report.json**:

```json
{
  "test_mae": 234.5,
  "test_rmse": 312.8,
  "test_mape": 42.3,
  "val_mae": 198.2,
  "val_rmse": 287.1,
  "val_mape": 38.7
}
```

### Entrenamiento del Modelo Baseline

El modelo baseline (Random Forest / XGBoost) se entrena mediante `main.py`:

```bash
python main.py --csv data/ALL_DATA.csv
```

Este modelo utiliza un enfoque diferente basado en features temporales y espaciales. Ver documentación en `main.py` para opciones adicionales.

### Búsqueda de Hiperparámetros

Para optimizar los hiperparámetros del modelo integrado, utiliza `grid_search.py`:

```bash
python grid_search.py
```

Este script prueba múltiples combinaciones de hiperparámetros y guarda la mejor configuración encontrada. Consulta el archivo para ver los rangos de búsqueda configurados.

### Ejemplos de Uso Avanzado

#### Modificar la Ventana Temporal

Edita `main_version_preliminar.py`:

```python
cfg.seq_len = 5  # Usa 5 años de historia en lugar de 3
```

#### Cambiar el Número de Vecinos en el Grafo

```python
cfg.gcn_k = 6  # Conecta con 6 vecinos más cercanos
```

#### Forzar Uso de CPU

```python
cfg.device = "cpu"
```

#### Ajustar Parámetros de Entrenamiento

```python
cfg.batch_size = 32      # Batch más pequeño para memoria limitada
cfg.epochs = 60          # Más épocas para entrenamiento más largo
cfg.lr = 5e-4            # Learning rate más conservador
cfg.hidden_dim = 128     # Dimensión oculta mayor
```

---

## Aplicación Web (Streamlit)

### Descripción

La aplicación web proporciona una interfaz interactiva para visualizar predicciones, explorar datos históricos y generar reportes. Está implementada con Streamlit y utiliza visualizaciones 3D mediante PyDeck.

### Acceder a la Aplicación

**Opción 1: Versión Desplegada (Recomendado)**

La aplicación está disponible en línea sin necesidad de instalación:

🔗 **[https://bladimiralfer-proyecto-cdd-app-ii1kxc.streamlit.app/](https://bladimiralfer-proyecto-cdd-app-ii1kxc.streamlit.app/)**

**Opción 2: Ejecución Local**

Para ejecutar la aplicación localmente en tu máquina:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador (por defecto en `http://localhost:8501`).

### Requisitos Previos

- El modelo debe estar entrenado y guardado en `ig_outputs/integrated_model.pt`
- Los archivos CSV de datos históricos deben estar en `data/`

### Funcionalidades Principales

#### 1. Panel de Control (Sidebar)

- **Filtros de Análisis**:
  - Rango de incidentes permitido: Filtra distritos por rango de predicciones
  - Filtro de distritos específicos: Selecciona uno o más distritos para análisis focalizado
  - Inclinación del mapa 3D: Ajusta el ángulo de visualización

#### 2. KPIs Tácticos

La aplicación muestra cuatro métricas clave:

- **Año de Proyección**: Año futuro para el cual se generan las predicciones
- **Proyección Total**: Suma total de delitos estimados (con delta porcentual vs año anterior)
- **Zona de Mayor Riesgo**: Distrito con mayor cantidad de delitos proyectados
- **Incidentes en Zona de Mayor Riesgo**: Volumen específico de delitos proyectados

Los KPIs se actualizan dinámicamente según los filtros aplicados (global o selección específica).

#### 3. Pestaña: Mapa Táctico

Visualización interactiva 3D de las predicciones:

- **Columnas 3D**: Altura proporcional a la cantidad de delitos proyectados
- **Colores dinámicos**: Escala de colores según nivel de riesgo
  - Gris oscuro: Bajo riesgo
  - Amarillo/Naranja: Medio riesgo
  - Rojo: Alto riesgo (Crítico)
- **Tooltips informativos**: Al hacer hover sobre una columna, se muestra:
  - Nombre del distrito
  - Proyección de delitos
  - Nivel de riesgo
  - Valor del año anterior
  - Tendencia (Sube/Baja)

**Leyenda**:
- Bajo: Incidentes < 500
- Medio: Incidentes entre 500-1500
- Crítico: Incidentes > 1500

#### 4. Pestaña: Comparativa

Gráfico de barras comparando:

- Valores reales del año anterior
- Predicciones para el año futuro

Muestra los top 20 distritos (o todos si hay filtro de selección activo). Permite identificar diferencias entre realidad y proyección para validar el modelo.

#### 5. Pestaña: Datos

Tabla interactiva con todos los datos operativos:

- Distrito
- Nivel de Riesgo
- Real Anterior (año pasado)
- Proyección (año futuro)
- Diferencia (absoluta)
- Tendencia (Sube/Baja)

**Funcionalidades**:
- Ordenamiento por cualquier columna
- Filtrado mediante selección de distritos en sidebar
- Gradiente de color en columna "Proyección" (rojo = mayor riesgo)
- Botón de descarga: Exporta los datos filtrados como CSV

### Arquitectura de la Aplicación

```
app.py
├── Carga del Sistema (@st.cache_resource)
│   ├── Carga modelo entrenado (integrated_model.pt)
│   ├── Carga nombres de distritos y configuración
│   ├── Carga scaler (mean, scale)
│   └── Carga matriz de adyacencia
│
├── Carga de Datos Históricos (@st.cache_data)
│   ├── Lee todos los CSV (final*.csv)
│   ├── Calcula centroides por distrito
│   └── Genera tabla pivot (año × distrito)
│
├── Inferencia
│   ├── Prepara ventana temporal (últimos seq_len años)
│   ├── Normaliza datos
│   ├── Ejecuta modelo (forward pass)
│   └── Desnormaliza predicciones
│
└── Visualización
    ├── KPIs (métricas principales)
    ├── Mapa 3D (PyDeck)
    ├── Gráfico comparativo (Altair)
    └── Tabla de datos (Pandas DataFrame)
```

### Optimizaciones de Rendimiento

- **Caché de recursos**: El modelo se carga una vez con `@st.cache_resource`
- **Caché de datos**: Los datos históricos se cargan una vez con `@st.cache_data`
- **Cálculos eficientes**: Las predicciones se calculan una vez al inicio y se reutilizan

### Personalización de la Interfaz

Los estilos CSS están definidos en la sección de configuración de `app.py`. Puedes modificar:

- Colores de métricas
- Estilos de leyenda
- Tema del mapa (dark/light)

---

## API y Funcionalidades

### Funciones Principales del Pipeline

#### `load_and_concatenate(glob_pattern: str) -> pd.DataFrame`

Carga y concatena múltiples archivos CSV.

**Parámetros**:
- `glob_pattern`: Patrón de búsqueda de archivos (ej: `"./data/final*.csv"`)

**Retorna**: DataFrame con todos los datos concatenados

**Lanza**: `FileNotFoundError` si no se encuentran archivos

#### `normalize_schema(df: pd.DataFrame) -> pd.DataFrame`

Normaliza y valida el esquema de datos.

**Parámetros**:
- `df`: DataFrame con datos crudos

**Retorna**: DataFrame normalizado con tipos correctos

**Lanza**: `ValueError` si faltan columnas requeridas

#### `build_district_centroids(df: pd.DataFrame) -> pd.DataFrame`

Calcula centroides geográficos por distrito.

**Parámetros**:
- `df`: DataFrame con columnas `X`, `Y`, `distrito`

**Retorna**: DataFrame con columnas `distrito`, `cent_x`, `cent_y`

#### `build_adjacency_from_centroids(centroids: pd.DataFrame, k: int = 4) -> Tuple[np.ndarray, List[str]]`

Construye matriz de adyacencia basada en k-vecinos más cercanos.

**Parámetros**:
- `centroids`: DataFrame con centroides
- `k`: Número de vecinos a conectar

**Retorna**: Tupla `(A_norm, district_names)` donde `A_norm` es matriz normalizada y `district_names` es lista ordenada de distritos

#### `aggregate_yearly_counts(df: pd.DataFrame, district_order: List[str]) -> pd.DataFrame`

Agrega conteos de delitos por año y distrito.

**Parámetros**:
- `df`: DataFrame con datos de incidentes
- `district_order`: Orden de distritos para columnas

**Retorna**: DataFrame pivot con años como índice y distritos como columnas

### Clases del Modelo

#### `IntegratedModel`

Modelo principal que integra todos los componentes.

**Parámetros del constructor**:
- `n_nodes`: Número de distritos (nodos)
- `seq_len`: Longitud de ventana temporal
- `in_ch`: Dimensión de entrada (1 para conteos)
- `hidden`: Dimensión oculta
- `A_norm`: Matriz de adyacencia normalizada

**Método forward**:
- `forward(seq: torch.Tensor) -> torch.Tensor`
  - Entrada: `(B, S, N)` donde B=batch, S=seq_len, N=n_nodes
  - Salida: `(B, N)` predicciones por distrito

### Uso Programático

Ejemplo de uso del modelo entrenado:

```python
import torch
from main_version_preliminar import IntegratedModel, CFG

# Cargar checkpoint
checkpoint = torch.load("ig_outputs/integrated_model.pt", map_location="cpu", weights_only=False)

# Extraer componentes
cfg_dict = checkpoint['cfg']
district_names = checkpoint['district_names']
A_norm = torch.tensor(checkpoint['A_norm'], dtype=torch.float32)
scaler_mean = checkpoint['scaler_mean']
scaler_scale = checkpoint['scaler_scale']

# Reconstruir modelo
model = IntegratedModel(
    n_nodes=len(district_names),
    seq_len=cfg_dict['seq_len'],
    in_ch=1,
    hidden=cfg_dict['hidden_dim'],
    A_norm=A_norm
)
model.load_state_dict(checkpoint['model'])
model.eval()

# Preparar datos (ejemplo: últimos 3 años)
# input_window debe ser array de shape (seq_len, n_districts)
input_scaled = (input_window - scaler_mean) / scaler_scale
tensor_in = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)  # Agregar dimensión batch

# Predecir
with torch.no_grad():
    pred_scaled = model(tensor_in).numpy()[0]

# Desnormalizar
pred_raw = (pred_scaled * scaler_scale) + scaler_mean
pred_raw = np.maximum(pred_raw, 0)  # Asegurar no negativos

# Resultado: pred_raw es array de shape (n_districts,)
```

---

## Deployment

### Opciones de Deployment

El sistema puede desplegarse en diferentes plataformas según los requisitos:

#### 1. Streamlit Cloud (Recomendado para MVP)

**Ventajas**:
- Deployment gratuito y sencillo
- Integración directa con GitHub
- Actualización automática con cada push

**Pasos**:
1. Sube el repositorio a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta el repositorio
4. Configura el comando: `streamlit run app.py`
5. Especifica el archivo principal: `app.py`

**Limitaciones**:
- Requiere que el modelo entrenado esté incluido en el repositorio (considera usar Git LFS para archivos grandes)
- Recursos limitados (CPU, memoria)

#### 2. Heroku

**Ventajas**:
- Control sobre recursos
- Configuración flexible

**Requisitos**:
- `Procfile`: `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- `runtime.txt`: Especifica versión de Python
- Variables de entorno para configuración

#### 3. Docker + Servidor Cloud

**Ventajas**:
- Máximo control y personalización
- Escalabilidad horizontal posible

**Ejemplo de Dockerfile**:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build y ejecución**:

```bash
docker build -t crime-prediction-app .
docker run -p 8501:8501 crime-prediction-app
```

#### 4. Servicios Cloud Empresariales

Para producción a gran escala, considera:
- **AWS**: EC2 + ECS/EKS, S3 para almacenamiento de modelos
- **Google Cloud**: Cloud Run, Vertex AI
- **Azure**: App Service, Container Instances

### URL de Deployment

**Aplicación Web Desplegada:**

🌐 **URL de Producción**: [https://bladimiralfer-proyecto-cdd-app-ii1kxc.streamlit.app/](https://bladimiralfer-proyecto-cdd-app-ii1kxc.streamlit.app/)

**Plataforma**: Streamlit Cloud

**Estado**: ✅ Activa y funcionando

La aplicación está desplegada en Streamlit Cloud y es accesible públicamente. Incluye todas las funcionalidades descritas en la sección [Aplicación Web (Streamlit)](#aplicación-web-streamlit), incluyendo:

- Visualización de predicciones en mapa 3D interactivo
- KPIs tácticos en tiempo real
- Comparativas históricas
- Exportación de datos
- Filtros avanzados por distrito y rango de incidentes

### Consideraciones para Producción

#### Seguridad

- **Autenticación**: Considera agregar autenticación para acceso restringido
- **Validación de entrada**: Valida todos los inputs del usuario
- **Rate limiting**: Limita número de requests por IP

#### Rendimiento

- **Caché de modelo**: El modelo se carga en memoria al iniciar la app
- **Optimización de queries**: Los datos históricos se cargan una vez y se cachean
- **CDN**: Para servir assets estáticos (si aplica)

#### Monitoreo

- **Logging**: Configura logging para errores y uso
- **Métricas**: Monitorea tiempo de respuesta, uso de memoria
- **Alertas**: Configura alertas para errores críticos

#### Actualización del Modelo

- **Versionado**: Mantén versiones del modelo con timestamps
- **A/B Testing**: Prueba nuevos modelos en producción con tráfico parcial
- **Rollback**: Plan de reversión si el nuevo modelo falla

---

## Métricas y Evaluación

### Métricas Principales

El sistema utiliza las siguientes métricas para evaluar el rendimiento:

#### 1. MAE (Mean Absolute Error)

**Definición**: Error absoluto medio entre predicciones y valores reales.

**Fórmula**: 
\[
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

**Interpretación**: Representa el error promedio en las mismas unidades que la variable objetivo (número de delitos). Un MAE de 200 significa que, en promedio, las predicciones se desvían 200 delitos del valor real.

**Uso**: Métrica principal para evaluar precisión absoluta. Más robusta a outliers que RMSE.

#### 2. RMSE (Root Mean Squared Error)

**Definición**: Raíz del error cuadrático medio.

**Fórmula**:
\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]

**Interpretación**: Penaliza más los errores grandes. Siempre mayor o igual que MAE. Un RMSE de 300 indica que los errores grandes tienen mayor peso en la evaluación.

**Uso**: Útil para identificar modelos con errores extremos. Usado como métrica de early stopping.

#### 3. MAPE (Mean Absolute Percentage Error)

**Definición**: Error porcentual absoluto medio.

**Fórmula**:
\[
MAPE = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
\]

**Interpretación**: Expresa el error como porcentaje del valor real. Un MAPE del 40% significa que, en promedio, las predicciones se desvían un 40% del valor real.

**Uso**: Útil para comparar rendimiento entre distritos con diferentes escalas de delitos. Solo calculado para valores no-cero.

**Limitación**: Puede ser problemático cuando los valores reales son cercanos a cero (división por valores pequeños).

### Interpretación de Resultados

#### Rendimiento Esperado

Para un modelo bien entrenado en datos de delitos urbanos:

- **MAE**: 150-300 delitos (depende del rango de valores)
- **RMSE**: 200-400 delitos
- **MAPE**: 30-50% (aceptable debido a variabilidad inherente de delitos)

#### Comparación entre Modelos

Al comparar el modelo integrado vs baseline:

| Métrica | Modelo Baseline | Modelo Integrado | Mejora |
|---------|----------------|------------------|--------|
| MAE     | [valor]        | [valor]          | [%]    |
| RMSE    | [valor]        | [valor]          | [%]    |
| MAPE    | [valor]        | [valor]          | [%]    |

**Criterios de evaluación**:
- **Mejora significativa**: Reducción ≥ 5% en MAE o RMSE
- **Mejora marginal**: Reducción < 5% pero > 0%
- **Sin mejora o regresión**: Aumento en métricas (requiere investigación)

### Validación Cruzada Temporal

Para evaluación más robusta, considera validación cruzada temporal:

1. **Walk-forward validation**: Entrena en [2016-2019], valida en 2020; luego entrena en [2016-2020], valida en 2021, etc.
2. **Expanding window**: Similar pero expandiendo la ventana de entrenamiento en cada iteración

**Implementación futura**: Script de validación cruzada temporal (ver `grid_search.py` como referencia).

### Análisis de Errores

Para entender mejor el comportamiento del modelo:

1. **Errores por distrito**: Identifica distritos con mayor error (puede indicar necesidad de más datos o features adicionales)
2. **Errores por rango de valores**: Evalúa si el modelo funciona mejor en distritos con muchos delitos vs pocos
3. **Errores temporales**: Verifica si hay sesgo en predicciones de años específicos

**Ejemplo de análisis**:

```python
# Después de entrenamiento
errors = abs(predictions - actuals)
high_error_districts = errors.nlargest(10).index
print(f"Distritos con mayor error: {high_error_districts}")
```

---

## Estructura del Proyecto

```
Proyecto_CDD/
│
├── data/                          # Datos de entrada
│   ├── final2016.csv              # Datos año 2016
│   ├── final2017.csv              # Datos año 2017
│   ├── ...
│   └── final2023.csv              # Datos año 2023
│
├── ig_outputs/                    # Salidas del modelo integrado
│   ├── integrated_model.pt        # Modelo entrenado (checkpoint)
│   └── report.json                # Reporte de métricas
│
├── outputs/                       # Salidas del modelo baseline
│   ├── metrics.json               # Métricas del modelo baseline
│   ├── feature_importance.csv     # Importancia de features
│   ├── predictions_2023.csv       # Predicciones año 2023
│   └── predictions_2024.csv       # Predicciones año 2024
│
├── app.py                         # Aplicación Streamlit (interfaz web)
│
├── main.py                        # Pipeline modelo baseline (RF/XGBoost)
│
├── main_version_preliminar.py     # Pipeline modelo integrado (GCN+LSTM+Attention)
│
├── grid_search.py                 # Búsqueda de hiperparámetros
│
├── modelando_MVP.ipynb            # Notebook de experimentación
│
├── hex_pordistrito.csv            # Mapeo de hexágonos H3 a distritos (opcional)
│
├── requirements.txt               # Dependencias Python
│
├── README.md                      # Este archivo
│
├── COMMITMENTS.md                 # Plan de trabajo y compromisos
│
└── PLAN.md                        # Plan estratégico y decisiones de diseño
```

### Descripción de Archivos Clave

#### `app.py`
Aplicación web interactiva con Streamlit. Requiere modelo entrenado en `ig_outputs/integrated_model.pt`.

#### `main_version_preliminar.py`
Pipeline completo del modelo integrado. Contiene:
- Carga y preprocesamiento de datos
- Construcción del grafo
- Definición del modelo
- Entrenamiento y evaluación
- Guardado de artefactos

#### `main.py`
Pipeline del modelo baseline. Utiliza enfoque basado en árboles con features temporales y espaciales.

#### `grid_search.py`
Script para optimización de hiperparámetros mediante búsqueda en grid o random search.

#### `requirements.txt`
Lista de todas las dependencias Python necesarias para el proyecto.

---

## Referencias

### Paper Principal

**Hou, X., et al.** (2022). "An Integrated Graph Model for Spatial–Temporal Urban Crime Prediction Based on Attention Mechanism". *[Revista/Conferencia]*.

Este paper proporciona la base teórica para la arquitectura del modelo integrado, específicamente:
- Uso de GCN para relaciones espaciales
- LSTM para dependencias temporales
- Mecanismo de atención para ponderación temporal

### Referencias Adicionales

1. **Cesario, E., et al.** (2024). "Multi-density crime predictor: an approach to forecast criminal high-risk areas in urban environments". *[Revista]*.

   Propone enfoque adaptativo para capturar patrones espaciales heterogéneos. Considerado para futuras mejoras.

2. **Kipf, T. N., & Welling, M.** (2017). "Semi-Supervised Classification with Graph Convolutional Networks". *ICLR*.

   Trabajo fundamental sobre GCN utilizado como base para la capa de convolución de grafos.

3. **Hochreiter, S., & Schmidhuber, J.** (1997). "Long Short-Term Memory". *Neural Computation*.

   Arquitectura LSTM original, base para el componente temporal del modelo.

4. **Vaswani, A., et al.** (2017). "Attention Is All You Need". *NeurIPS*.

   Introducción del mecanismo de atención, adaptado para atención temporal en este proyecto.

### Datos y Recursos

- **Datos de delitos**: Fuente gubernamental (especificar si es público o privado)
- **Coordenadas geográficas**: Sistema de coordenadas utilizado (ej: WGS84, UTM)
- **Límites administrativos**: Si se utilizan en futuras versiones (fuente: [especificar])

### Herramientas y Librerías

- **PyTorch**: Framework de deep learning ([pytorch.org](https://pytorch.org/))
- **Streamlit**: Framework para aplicaciones web en Python ([streamlit.io](https://streamlit.io/))
- **PyDeck**: Visualización de mapas 3D ([pydeck.gl](https://pydeck.gl/))
- **scikit-learn**: Herramientas de ML tradicional ([scikit-learn.org](https://scikit-learn.org/))

---

## Solución de Problemas

### Errores Comunes y Soluciones

#### Error: "No se encontraron CSV con el patrón"

**Causa**: Los archivos CSV no están en la ubicación esperada o no siguen el patrón de nombres.

**Solución**:
```bash
# Verificar que los archivos existen
ls data/final*.csv

# Verificar el patrón en el código
# En main_version_preliminar.py, verifica:
cfg.data_glob = "./data/final*.csv"  # Ajusta la ruta si es necesario
```

#### Error: "La serie tiene solo X pasos"

**Causa**: No hay suficientes años de datos para crear ventanas temporales de longitud `seq_len`.

**Solución**:
```python
# Reduce seq_len en la configuración
cfg.seq_len = 2  # En lugar de 3

# O verifica que tienes suficientes años
# Necesitas al menos seq_len + 1 años (ej: seq_len=3 requiere mínimo 4 años)
```

#### Error: "Falta columna requerida"

**Causa**: Los CSV no tienen todas las columnas necesarias.

**Solución**:
Verifica que cada CSV tenga estas columnas:
- `anio` (numérico)
- `X` (numérico, longitud)
- `Y` (numérico, latitud)
- `distrito` (texto)

Usa pandas para inspeccionar:
```python
import pandas as pd
df = pd.read_csv("data/final2016.csv")
print(df.columns.tolist())
```

#### Error: "CUDA out of memory"

**Causa**: El modelo es demasiado grande para la GPU disponible, o el batch_size es muy grande.

**Solución**:
```python
# Opción 1: Reducir batch_size
cfg.batch_size = 32  # O 16, o 8

# Opción 2: Reducir hidden_dim
cfg.hidden_dim = 32  # En lugar de 64

# Opción 3: Usar CPU
cfg.device = "cpu"
```

#### Error: "Modelo no converge" (pérdida no disminuye o NaN)

**Causa**: Learning rate muy alto, datos no normalizados correctamente, o gradientes explotando.

**Solución**:
```python
# Reducir learning rate
cfg.lr = 5e-4  # O 1e-4

# Verificar normalización de datos
# Asegúrate de que el scaler se ajusta solo con datos de entrenamiento

# Agregar gradient clipping (si no está ya)
# En el loop de entrenamiento:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### Error: Streamlit no carga el modelo

**Causa**: El archivo `ig_outputs/integrated_model.pt` no existe o está corrupto.

**Solución**:
```bash
# Verificar que el modelo existe
ls -lh ig_outputs/integrated_model.pt

# Si no existe, entrena primero:
python main_version_preliminar.py

# Verificar que el checkpoint es válido
python -c "import torch; torch.load('ig_outputs/integrated_model.pt', map_location='cpu')"
```

#### Error: "Distritos no coinciden" en Streamlit

**Causa**: Los nombres de distritos en los CSV no coinciden exactamente con los del modelo entrenado.

**Solución**:
- Verifica normalización de nombres (mayúsculas, espacios)
- El pipeline normaliza nombres a mayúsculas y elimina espacios
- Asegúrate de usar los mismos datos de entrenamiento y predicción

### Optimización de Rendimiento

#### Entrenamiento muy lento

**Soluciones**:
1. Usar GPU si está disponible
2. Reducir `batch_size` solo si hay problemas de memoria (batch más grande suele ser más rápido)
3. Reducir `hidden_dim` o `seq_len`
4. Reducir número de épocas (usar early stopping)

#### Aplicación Streamlit lenta

**Soluciones**:
1. Verificar que `@st.cache_resource` y `@st.cache_data` están siendo usados
2. Reducir cantidad de distritos mostrados en el mapa
3. Simplificar visualizaciones (menos datos en gráficos)
4. Usar datos muestreados para visualización rápida

### Depuración

#### Logging detallado

Agrega logging para entender el flujo:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# En funciones clave:
logger.debug(f"Cargando {len(files)} archivos CSV")
logger.info(f"Entrenando con {len(train_dataset)} ejemplos")
```

#### Verificar dimensiones de tensores

Agrega prints temporales para verificar shapes:

```python
# En el forward del modelo
print(f"Input shape: {x.shape}")
x = self.resblock(x)
print(f"After resblock: {x.shape}")
# ... etc
```

#### Validar datos de entrada

```python
# Verificar que no hay NaN o infinitos
assert not np.isnan(input_window).any(), "NaN en datos de entrada"
assert not np.isinf(input_window).any(), "Inf en datos de entrada"
```





