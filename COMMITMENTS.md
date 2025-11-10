# Compromisos y Plan de Trabajo Final

## Proyecto: Predicción Espacio-Temporal de Delitos en Lima Metropolitana

Duración del plan: **2 (10–24 de noviembre de 2025)**  

---

## Objetivo General

Optimizar y documentar completamente el pipeline de predicción de delitos en Lima Metropolitana, incorporando variables contextuales y simulaciones, asegurando reproducibilidad, mejora de métricas y preparación del modelo para su posible despliegue o extensión futura.

---

## 🧩 Objetivos SMART

| Nº | Objetivo SMART | Responsable | Fecha límite | Resultado esperado |
|----|----------------|--------------|---------------|--------------------|
| 1 | **Refinar el dataset** integrando variables contextuales sintéticas (clima, densidad poblacional, cobertura policial). | Bladimir | 15 nov 2025 | Nuevas columnas agregadas y documentadas en el notebook. |
| 2 | **Reentrenar el pipeline** con las variables nuevas y comparar desempeño con el modelo base (ΔR² o MAE). | Martin | 17 nov 2025 | `metrics.json` actualizado con mejoras cuantificables. |
| 3 | **Simular escenarios de predicción para 2024**, generando una función `predict_new_data()` para futuros registros. | Stuart | 20 nov 2025 | Archivo `predictions_2024.csv` validado y código funcional. |
| 4 | **Documentar las mejoras, resultados y limitaciones** en el `README.md` y `COMMITMENTS.md`. | Bladimir | 22 nov 2025 | Documentación final actualizada en GitHub. |
| 5 | **Validar reproducibilidad del proyecto** ejecutando el pipeline completo en un entorno limpio. | Todos | 24 nov 2025 | Ejecución exitosa sin errores con el comando estándar. |

---

## Métricas de Éxito

| Métrica | Descripción | Meta |
|----------|--------------|------|
| **ΔR²** | Incremento en el coeficiente de determinación respecto al modelo base | ≥ +0.05 |
| **MAE** | Error absoluto medio | < 10 |
| **MAPE (%)** | Error porcentual medio absoluto | < 25% |
| **Reproducibilidad** | Ejecución exitosa en entorno nuevo | 100% |
| **Documentación** | README y COMMITMENTS completos y claros | 100% |

---

## Plan de Trabajo 

| Semana | Tareas principales | Responsables | Entregables |
|---------|--------------------|---------------|--------------|
| **Semana 1 (10–17 nov)** | - Agregar variables contextuales.<br>- Ajustar y limpiar el dataset consolidado.<br>- Reentrenar el modelo y analizar resultados. | Martín y Bladimir | Dataset actualizado (`ALL_DATA.csv`) y `metrics.json` con comparación. |
| **Semana 2 (18–24 nov)** | - Implementar predicción .<br>- Validar el pipeline completo en entorno limpio.<br>- Actualizar documentación y repositorio final. | Stuart y todos | Predicciones (`predictions.csv`) y repo final documentado. |

---

## ⚙️ Plan B

| Riesgo | Estrategia alternativa |
|---------|------------------------|
| Variables contextuales difíciles de obtener | Generar datos sintéticos con distribuciones basadas en densidad de delitos y cobertura policial histórica. |
| Métricas no mejoran significativamente | Ajustar hiperparámetros (`max_depth`, `n_estimators`) o cambiar a XGBoost. |
| Incompatibilidad de librerías o errores en H3 | Usar agrupación por distrito como fallback y guardar coordenadas originales. |
