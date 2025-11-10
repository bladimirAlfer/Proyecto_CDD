# Predicción Espacio-Temporal de Delitos en Lima Metropolitana

---

## Descripción General

Este proyecto desarrolla un **pipeline reproducible de machine learning** para predecir la **tasa de delitos por zona (H3)** en **Lima Metropolitana**, usando datos anuales de denuncias policiales del 2016 al 2023.  
El objetivo es explorar patrones espacio-temporales de criminalidad y generar **predicciones futuras** (por ejemplo, para el año 2024).

El sistema integra procesamiento geoespacial, ingeniería de características temporales y entrenamiento supervisado con **Random Forest** o **XGBoost**, de forma totalmente automatizada y documentada.

---

## Requisitos

### Pre-requisitos

1. **Python 3.10 o superior**

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

## Ejecución del Notebook

Asegurar de que los archivos final2016.csv a final2023.csv estén en la carpeta data/.
El script del notebook modelando_MVP.ipynb leerá los datasets y generará los archivos necesarios.

## Ejecución del Pipeline

Para entrenar y evaluar el modelo con el dataset consolidado, ejecuta el siguiente comando desde la raíz del proyecto:  

```bash
python main.py --csv data/ALL_DATA.csv --train-end 2021 --val-year 2022 --test-year 2023
```

