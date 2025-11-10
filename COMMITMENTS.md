# Compromisos y Plan de Trabajo Final

## Proyecto: Predicción Espacio-Temporal de Delitos en Lima Metropolitana

Duración del plan: **2 (10–24 de noviembre de 2025)**  

---

## Objetivo General

Consolidar y preparar el MVP del proyecto para su presentación final, priorizando:
- Validación técnica y comparativa del modelo integrado vs. baseline.
- Simulación de despliegue (predicciones 2024–2025).
- Documentación extendida, interpretación de resultados y comunicación visual.
- Reproducibilidad total y empaquetado final para evaluación.

---

## 🧩 Objetivos SMART

| Nº | Tarea SMART | Responsable | Fecha límite | Métrica de Éxito |
|----|--------------|--------------|---------------|------------------|
| 1 | **Refinar la simulación de predicción 2024–2025**, generando escenarios alternativos (variación espacial + temporal). | Bladimir | 15 nov 2025 | Predicciones guardadas (`predictions_2024.csv`, `predictions_2025.csv`) sin errores de ejecución. |
| 2 | **Implementar análisis visual y narrativo** de resultados (mapas de calor, error por distrito, tendencia temporal). | Martín | 17 nov 2025 | 3 visualizaciones exportadas (`outputs/visuals/`), notebook actualizado. |
| 3 | **Empaquetar el pipeline completo** (baseline + modelo avanzado) en un solo flujo reproducible (`run_all.py`). | Stuart | 20 nov 2025 | Script ejecutable que reproduce todas las etapas en <5 min de ejecución. |
| 4 | **Preparar documentación final y MVP.** | Todos | 22 nov 2025 | README extendido con explicación técnica. |
| 5 | **Simular despliegue local del modelo** con input manual de coordenadas/distrito. | Bladimir | 24 nov 2025 | Predicción en consola o mapa interactivo funcional (MVP demostrable). |

---

## Métricas de Éxito

| Métrica | Descripción | Meta |
|----------|--------------|------|
| **MAE ≤ 10** | Error medio absoluto en test final | Cumplido o mejorado |
| **RMSE ≤ 15** | Error cuadrático medio en test final | Cumplido |
| **ΔMAE (↓)** | Reducción ≥ 10% respecto al baseline | Cumplido |
| **Reproducibilidad** | Ejecución completa del proyecto en entorno limpio | 100% |
| **Documentación** | README y COMMITMENTS completos, claros y actualizados | 100% |


---

## ⚙️ Plan B — Estrategias ante fallos críticos

| Componente Crítico | Riesgo Potencial | Estrategia de Plan B |
|--------------------|------------------|----------------------|
| **Codificación espacial (H3)** | Error en librería `h3` o coordenadas inválidas. | Sustituir H3 por agrupación por **distrito** o cuadrantes definidos manualmente (`grid_id`). Entrenar el modelo solo con tasas por distrito. |
| **Escalado y normalización** | `StandardScaler` genera NaN o valores extremos. | Implementar control de errores (`np.isnan`) y usar **MinMaxScaler** por distrito si es necesario. |
| **Tiempo de entrenamiento excesivo** | Dataset amplio o falta de hardware. | Reducir cantidad de distritos a los 10 más incidentes y disminuir `epochs` y `hidden_dim`. |

---

## 📈 Estado Actual vs Estado Esperado (10–24 nov 2025)

| Componente | Estado Actual | Estado Esperado (24 nov 2025) |
|-------------|----------------|-------------------------------|
| **Dataset consolidado (`ALL_DATA.csv`)** | Validado y unificado. | Enriquecido con variables contextuales (cobertura, patrullaje). |
| **Modelo avanzado (GCN + LSTM + Attention)** | Entrenado y probado |Optimizado y validado, guardado en `ig_outputs/integrated_model.pt`. |
| **Predicciones futuras (2024–2025)** | Parcialmente generadas. | Extendidas, validadas y almacenadas en `/outputs/`. |
| **Documentación (README / COMMITMENTS)** | Actualizada parcialmente. | Finalizada con guía de ejecución y análisis comparativo. |
| **Simulación de Deployment Local** | No implementada. | Ejecutable con predicción puntual por distrito o coordenadas. |

---

## Criterios de Deployment si el MVP resulta funcional

| Criterio | Descripción | Resultado Esperado |
|-----------|--------------|--------------------|
| **Ejecución Reproducible End-to-End** | El proyecto debe correr con un solo comando que ejecute todo el flujo (baseline + modelo avanzado). | `python run_all.py --csv data/ALL_DATA.csv` genera predicciones, métricas y visualizaciones sin errores. |
| **Simulación de Despliegue Local** | Debe poder ejecutarse una predicción manual ingresando un distrito o coordenadas. | `python run_all.py --predict "CHORRILLOS,-12.17,-77.02"` devuelve predicción estimada. |



