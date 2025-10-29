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

**Clasificación:** Inversión Estratégica 

**Justificación:**  

Esta iniciativa tiene **alto impacto en el producto** y **costo técnico medio-alto**, ya que implica reestructurar parcialmente el pipeline de ML, y en la calidad de los datos de entrada, lo que implica **un esfuerzo técnico intermedio**, pero con **alto impacto en la precisión y confianza del sistema**.  

Abordar el gap identificado:
- Resuelve el “cuello de botella” de precisión en zonas de baja frecuencia, habilitando una funcionalidad nueva: predicción confiable en áreas subrepresentadas.
- Implica reentrenar con técnicas de balanceo de clases o generación sintética (SMOTE, oversampling).
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

## 2. Solución Técnica Propuesta para el Componente ML

### Técnica/Paper Seleccionado #1  
**Título:** *Multi-density crime predictor: an approach to forecast criminal high-risk areas in urban environments* (Cesario et al., 2024)  
**Principal contribución:**  
El artículo propone un enfoque predictivo denominado **MD-CrimePredictor**, que combina análisis espacial de densidades múltiples con modelos temporales (SARIMA y LSTM) para anticipar zonas urbanas de alto riesgo criminal. A diferencia de los métodos tradicionales que usan rejillas fijas o áreas administrativas, este enfoque adapta dinámicamente el tamaño y la forma de las zonas según la densidad de delitos, permitiendo capturar patrones espaciales heterogéneos y mejorar la precisión en la identificación de *hotspots*.  

**Justificación clave del triage:**  
Se selecciona este enfoque por su **relevancia directa con el objetivo del proyecto**, al abordar la predicción espacio-temporal de criminalidad considerando la variabilidad espacial de las zonas urbanas. Además, su estructura modular (detección de densidades + modelado temporal) puede integrarse fácilmente en nuestro *pipeline* hexagonal existente, reemplazando la etapa de agregación fija por un agrupamiento adaptativo de hexágonos con alta concentración delictiva. Esto maximiza la factibilidad de implementación sin requerir arquitecturas complejas de *deep learning*, manteniendo interpretabilidad y compatibilidad con los datos disponibles (2016–2023).  

---

### Técnica/Paper Seleccionado #2  
**Título:** *An Integrated Graph Model for Spatial–Temporal Urban Crime Prediction Based on Attention Mechanism* (Hou et al., 2022)  
**Principal contribución:**  
Este trabajo introduce un modelo **espacio-temporal basado en grafos y mecanismos de atención**, combinando redes convolucionales de grafo (*GCN*) con modelos secuenciales (*LSTM*) para capturar dependencias tanto espaciales (entre distritos o zonas vecinas) como temporales (tendencias históricas de crimen). El mecanismo de atención permite ponderar dinámicamente la influencia de áreas vecinas y periodos previos, mejorando la precisión predictiva y la interpretabilidad.  

**Justificación clave del triage:**  
Se elige esta técnica por ofrecer una **base conceptual avanzada** para extender nuestro modelo actual. Su énfasis en la **relación entre celdas adyacentes** (análogas a los *neighbors* H3) resulta directamente aplicable a nuestro enfoque hexagonal, permitiendo incorporar la influencia espacial de hexágonos vecinos mediante agregaciones o grafos H3. Aunque su implementación completa requeriría infraestructura de *deep learning*, sus principios pueden integrarse progresivamente en nuestro sistema (por ejemplo, añadiendo *features* de tasa promedio de vecinos o pesos espaciales basados en atención), brindando una evolución natural hacia un modelo más robusto y contextual.  

## 3. Hipótesis SMART y Métricas Principales

**Hipótesis SMART:**
> Al integrar el componente híbrido MD-CrimePredictor + GCN-Attention en nuestro pipeline de predicción, esperamos **incrementar el F1-Score en la clase minoritaria (zonas de baja incidencia delictiva) en un 20%** (pasando de 0.61 a >0.73) y **reducir el error espacial promedio (MAPE espacial)** en un 25% (de 53% a <40%) en el dataset de validación.  
> Esto se traducirá en una **mejora del 15% en la eficiencia de asignación de patrullaje** en los distritos piloto durante las próximas 4 semanas de pruebas.

---

### Métricas Principales de Éxito

- **Métrica Técnica Principal:**  
  - *F1-Score en zonas de baja incidencia delictiva*  
    *(Meta: incremento ≥ 20% respecto al baseline Random Forest).*  
  - Complementaria: *MAPE espacial* en celdas periféricas (meta: reducción ≥ 25%).

### Métrica de Negocio/Producto Principal

**Reducción porcentual de delitos reportados en hotspots predichos (normalizada por población, evaluación anual).**  
Dado que la predicción y el despliegue se evalúan a nivel anual, esta métrica compara el promedio anual de delitos en las zonas identificadas como hotspots en el periodo baseline (promedio de los últimos N años pre-intervención, por ejemplo 3 años) contra el año del piloto o el primer año post-despliegue.

**Definición formal:**  

$$\text{Reducción \% anual} = 100 \times \frac{\text{Crímenes}_{\text{baseline\_anual}} - \text{Crímenes}_{\text{post\_anual}}}{\text{Crímenes}_{\text{baseline\_anual}}}$$


$\text{Crímenes}_{\text{baseline\_anual}}$: promedio anual de delitos en el/los hotspot(s) durante los N años previos al piloto (recomendado N=3 para suavizar variaciones).  
$\text{Crímenes}_{\text{post\_anual}}$: cantidad de delitos en el año del piloto o primer año post-despliegue.

**Normalización por población:** para comparar zonas con distintas poblaciones se calcula también delitos por 10,000 habitantes:

$$\text{Delitos por 10k (anual)} = \frac{\text{Delitos anuales}}{\text{Población}} \times 10000$$


**Meta sugerida:** ≥ 12% reducción en delitos por 10k habitantes en hotspots predichos durante el primer año post-despliegue (o mejora estadísticamente significativa frente a distritos control).
