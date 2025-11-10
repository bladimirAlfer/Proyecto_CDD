# Compromisos y Plan de Trabajo

## Proyecto: Predicción Espacio-Temporal de Delitos en Lima Metropolitana

**Fecha de inicio:** 10 de noviembre de 2025  
**Duración del plan:** 2 semanas (10–23 de noviembre de 2025)

---

## Objetivo General

Consolidar y mejorar el MVP del proyecto, enfocándose en:
- **Mejoras del frontend**: Optimización de la aplicación Streamlit para mejor usabilidad y visualización
- **Búsqueda de hiperparámetros**: Optimización del modelo integrado (GCN + LSTM + Attention) mediante grid search o random search
- **Validación comparativa**: Comparación entre modelo baseline y modelo integrado
- **Documentación actualizada**: README y COMMITMENTS reflejando el estado actual del proyecto

---

## 🧩 Tareas Específicas (Próximas 2 Semanas)

| Nº | Tarea | Fecha límite | Métrica de Éxito Cuantificable |
|----|-------|--------------|--------------------------------|
| 1 | **Mejorar interfaz de Streamlit (app.py)** | 13 nov 2025 | 3 mejoras implementadas: (1) Filtros por rango de valores, (2) Comparación lado a lado de años, (3) Exportación de gráficos |
| 2 | **Implementar búsqueda de hiperparámetros para modelo integrado** | 16 nov 2025 | Script de grid search ejecutado, al menos 20 combinaciones probadas, mejor configuración guardada con métricas |
| 3 | **Optimizar visualizaciones del mapa** | 18 nov 2025 | Tiempo de carga < 3 segundos, tooltips informativos, leyenda clara |
| 4 | **Comparar modelos baseline vs integrado** | 20 nov 2025 | Tabla comparativa generada con métricas (MAE, RMSE, MAPE) para ambos modelos |
| 5 | **Actualizar documentación (README y COMMITMENTS)** | 22 nov 2025 | README actualizado con instrucciones del modelo integrado, COMMITMENTS con estado actual vs esperado |
| 6 | **Validación final y pruebas** | 23 nov 2025 | Pipeline completo ejecutable sin errores, métricas documentadas, frontend funcional |

---

## Métricas de Éxito Cuantificables

| Métrica | Descripción | Meta Actual | Meta Esperada (23 nov) |
|----------|-------------|-------------|------------------------|
| **MAE (Modelo Integrado)** | Error medio absoluto en test | Baseline actual | Reducción ≥ 5% vs baseline |
| **RMSE (Modelo Integrado)** | Error cuadrático medio en test | Baseline actual | Reducción ≥ 5% vs baseline |
| **Tiempo de carga (Frontend)** | Tiempo para cargar mapa en Streamlit | Actual | < 3 segundos |
| **Cobertura de hiperparámetros** | Combinaciones probadas en grid search | 0 | ≥ 20 combinaciones |
| **Usabilidad del frontend** | Funcionalidades nuevas implementadas | Actual | +3 mejoras (filtros, comparación, exportación) |
| **Reproducibilidad** | Ejecución completa sin errores | Parcial | 100% (ambos modelos) |

---

## ⚙️ Plan B — Estrategias ante Fallos Críticos

| Componente Crítico | Riesgo Potencial | Estrategia de Plan B |
|--------------------|------------------|----------------------|
| **Búsqueda de hiperparámetros muy lenta** | Grid search toma > 24 horas | Reducir espacio de búsqueda a 5-10 combinaciones más prometedoras, usar random search en lugar de grid search completo |
| **Modelo integrado no converge** | Pérdida no disminuye o NaN durante entrenamiento | Reducir learning rate, disminuir `hidden_dim` a 32, reducir `epochs` a 20, verificar normalización de datos |
| **Frontend lento o con errores** | Streamlit crashea o tarda mucho en cargar | Simplificar visualizaciones, usar cache más agresivo (`@st.cache_data`), reducir tamaño de datos mostrados, deshabilitar capas opcionales |
| **Dependencias faltantes** | PyTorch o librerías no instaladas | Documentar instalación paso a paso, crear script de setup automático, usar entornos virtuales |
| **Datos incompletos o corruptos** | Errores al cargar CSVs | Validar datos antes de procesar, agregar manejo de errores, usar datos de ejemplo si faltan archivos |

---

## 📈 Estado Actual vs Estado Esperado (9–23 nov 2025)

| Componente | Estado Actual (9 nov 2025) | Estado Esperado (23 nov 2025) |
|-------------|----------------------------|-------------------------------|
| **Modelo Baseline (main.py)** | ✅ Funcional con Random Forest/XGBoost, genera predicciones 2023-2024 | ✅ Mantener funcional, documentado |
| **Modelo Integrado (main_version_preliminar.py)** | ✅ Implementado con GCN+LSTM+Attention, hiperparámetros por defecto | ✅ Optimizado con mejor configuración encontrada mediante búsqueda |
| **Frontend (app.py)** | ✅ Básico funcional con mapas Pydeck y Folium | ✅ Mejorado con filtros, comparaciones y exportación de gráficos |
| **Búsqueda de hiperparámetros** | ❌ No implementada | ✅ Script funcional que prueba múltiples configuraciones y guarda mejores resultados |
| **Comparación de modelos** | ⚠️ Parcial (solo métricas individuales) | ✅ Tabla comparativa completa con métricas de ambos modelos |
| **Documentación** | ⚠️ README desactualizado, COMMITMENTS con fechas antiguas | ✅ README actualizado con ambos modelos, COMMITMENTS con plan actual |

---

## Criterios de Deployment si el MVP resulta funcional

| Criterio | Descripción | Resultado Esperado |
|-----------|-------------|--------------------|
| **Ejecución Reproducible** | Ambos modelos deben ejecutarse sin errores en entorno limpio | `python main.py --csv data/ALL_DATA.csv` y `python main_version_preliminar.py` funcionan correctamente |
| **Frontend Funcional** | La aplicación Streamlit debe cargar y mostrar predicciones | `streamlit run app.py` inicia sin errores, muestra mapas y métricas correctamente |
| **Métricas Documentadas** | Todas las métricas deben estar guardadas y accesibles | Archivos `outputs/metrics.json` y `ig_outputs/report.json` generados con métricas completas |
| **Hiperparámetros Optimizados** | Mejor configuración encontrada y documentada | Archivo con mejores hiperparámetros guardado, mejora de métricas documentada |
| **Comparación de Modelos** | Tabla o reporte comparando ambos enfoques | Documento o tabla mostrando ventajas/desventajas de cada modelo |

---

## Notas Adicionales

- **Prioridad**: Las mejoras del frontend y la búsqueda de hiperparámetros son el foco principal
- **Flexibilidad**: Si alguna tarea toma más tiempo del esperado, priorizar funcionalidad sobre perfección
- **Comunicación**: Actualizar COMMITMENTS.md semanalmente con progreso real

