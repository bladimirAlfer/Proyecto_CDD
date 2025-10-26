## 1. Diagnóstico del Gap de ML y Decisión Estratégica

### Descripción del Gap Crítico  
El modelo actual de predicción de delitos, basado en **Random Forest**, presenta un desempeño sólido con un **R² = 0.90** y un **MAE = 15.9**, lo que confirma su capacidad para identificar correctamente las zonas críticas de alta incidencia delictiva.  

Sin embargo, presenta un **MAPE del 53 %**, lo cual revela un **gap de precisión relativo en zonas de baja densidad delictiva o con datos incompletos**.  
En estas áreas, el modelo tiende a **sobrestimar o subestimar la criminalidad**, afectando la confiabilidad del sistema para planificar acciones preventivas en distritos periféricos o con menor volumen de denuncias.  

Este gap limita la posibilidad de escalar el MVP hacia un **sistema de alerta temprana o planificación automática de patrullaje**, donde la exactitud en zonas de baja frecuencia delictiva es esencial.  

---

### Matriz de Decisión  


| **Tipo de Iniciativa ** | **Impacto en el Producto** | **Costo / Riesgo Técnico** | **Ejemplo** | **Acción Recomendada** |
|------------------------------------------|----------------------------|-----------------------------|----------------------------------|-------------------------|
| **Quick Wins** | Alto impacto, bajo costo. | Bajo riesgo. | Mejoras visuales, tooltips, ajustes de mapa y colores para mayor usabilidad. | **Priorizar.** |
| **Inversión Estratégica** | Alto impacto, alto costo. | Riesgo medio-alto (requiere rediseño de pipeline). | Reentrenar el modelo con datos balanceados, calibración probabilística y nuevas variables contextuales. | **Planificar y justificar.**  |
| **Experimentos / Aprendizaje** | Bajo impacto, bajo costo. | Bajo riesgo. | Probar nuevas técnicas de muestreo o pequeños cambios en hiperparámetros. | **Time-box / Hackathon.** |
| **Distracción Costosa (Trampas)** | Bajo impacto, alto costo. | Alto riesgo. | Migrar a arquitecturas complejas (redes neuronales espaciales) sin suficientes datos ni infraestructura. | **Evitar.** |


---

### Clasificación y Justificación  
**Clasificación:** *Inversión Estratégica*  

**Justificación:**  

Esta iniciativa tiene **alto impacto en el producto** y **costo técnico medio-alto**, ya que implica reestructurar parcialmente el pipeline de ML, y en la calidad de los datos de entrada, lo que implica **un esfuerzo técnico intermedio**, pero con **alto impacto en la precisión y confianza del sistema**.  

Abordar el gap identificado:
- Resuelve el “cuello de botella” de precisión en zonas de baja frecuencia, habilitando una funcionalidad nueva: predicción confiable en áreas subrepresentadas.
- Implica reentrenar con técnicas de balanceo de clases o generación sintética (SMOTE, oversampling)**.
- Aplica calibración probabilística o cuantificación de incertidumbre para mejorar la interpretación de las predicciones.  
- Integra variables contextuales externas (iluminación pública, densidad poblacional, tipo de vía, proximidad a mercados o comisarías).  

Aunque implica mayor esfuerzo técnico, esta inversión mejora la calibración del modelo y la calidad de datos, y asi asegurar que el sistema sea confiable y escalable.  


---

### Evaluación de Costo y Riesgo (Eje X)  

| **Dimensión** | **Evaluación** | **Descripción** |
|---------------|----------------|-----------------|
| **Costo de Datos** | Medio | Se necesitan variables adicionales (población, infraestructura, etc.) pero accesibles en fuentes abiertas (INEI, Geoportal MML). |
| **Costo de Cómputo** | Medio | El modelo se puede reentrenar localmente; no requiere GPU ni entrenamiento a gran escala. |
| **Costo de Expertise** | Medio | Requiere conocimiento en calibración y feature engineering, pero el equipo actual puede implementarlo. |
| **Costo de Monitoreo** | Bajo | El modelo puede seguir monitoreado con métricas estándar (MAE, R², MAPE). |

> **Conclusión:** Riesgo controlado y costo asumible para el valor que genera en el producto.

---

### Evaluación de Impacto (Eje Y)  

| **Tipo de Impacto** | **Nivel** | **Evidencia o Razón** |
|----------------------|-----------|-----------------------|
| **Impacto en el producto** | Alto | Mejora la capacidad predictiva en zonas críticas (nuevo valor para usuario). |
| **Impacto en el usuario final (PNP / Serenazgo)** | Alto | Permite decisiones más precisas sobre dónde asignar patrullas o cámaras. |
| **Impacto en el negocio / escalabilidad** | Medio-Alto | Mejora la confiabilidad del sistema y habilita fases futuras (alertas en tiempo real). |
| **Reducción de riesgo (sesgo, error, falta de datos)** | Alto | Corrige el sesgo actual hacia zonas con más denuncias. |

> **Conclusión:** El impacto no solo mejora una métrica, sino que reduce un riesgo estructural y aumenta el valor operativo del sistema.

 
