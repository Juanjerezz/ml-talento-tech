# Detección de Fraude en Transacciones con Tarjeta de Crédito

Proyecto final del curso de **Machine Learning — Talento Tech**.

Pipeline completo de Machine Learning para detectar transacciones fraudulentas con tarjeta de crédito, entrenado sobre un dataset real con un **desbalance extremo de clases** (0,17% de fraude), abordando de punta a punta los desafíos propios de este tipo de problemas: preparación de datos sin fuga de información, elección de métricas adecuadas para clases desbalanceadas y selección justificada del modelo final.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/status-completado-brightgreen)
![Colab](https://img.shields.io/badge/Google%20Colab-ejecutado-yellow)

---

## Descripción

El fraude financiero genera pérdidas económicas directas y afecta la confianza de los clientes. Este proyecto aborda el problema como una tarea de **clasificación binaria supervisada**: dado un conjunto de variables de una transacción, predecir si se trata de una operación legítima o fraudulenta.

El trabajo cubre el flujo completo de un proyecto de ciencia de datos aplicado: exploración inicial, limpieza, análisis de correlaciones, prevención de *data leakage*, entrenamiento y comparación de modelos, evaluación bajo métricas apropiadas para clases desbalanceadas, validación cruzada estratificada y selección justificada del modelo final.

## Objetivo

Construir un modelo de Machine Learning capaz de identificar transacciones fraudulentas con la mayor precisión y recall posibles, priorizando un sistema de alertas que sea **operativamente viable** (es decir, que no genere un volumen de falsos positivos imposible de revisar en la práctica).

## Dataset

- **Fuente:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Machine Learning Group, ULB (Kaggle)
- **Transacciones:** 284.807 (283.726 tras eliminar 1.081 duplicados)
- **Fraudes:** 492 casos (~0,17% del total → desbalance extremo)
- **Variables:** `Time`, `Amount` en escala original + `V1`–`V28`, resultado de una transformación PCA aplicada por los autores del dataset para proteger la confidencialidad de la información original
- **Valores faltantes:** ninguno

## Metodología

1. **Exploración inicial:** estructura del dataset, tipos de datos, estadísticas generales.
2. **Limpieza:** verificación de nulos y eliminación de duplicados.
3. **Análisis exploratorio:** distribución de la variable objetivo, distribución de `Amount`/`Time`, detección de outliers y mapa de correlaciones.
4. **División del dataset:** `train_test_split` 80/20 con `stratify=y` para mantener la proporción de fraude en ambos conjuntos.
5. **Escalado sin fuga de información:** `StandardScaler` ajustado **únicamente** con `X_train`, aplicado luego a `X_test`.
6. **Manejo del desbalance:** `class_weight="balanced"` en ambos modelos, en lugar de sobremuestreo/submuestreo.
7. **Entrenamiento de dos modelos:** Regresión Logística y Random Forest.
8. **Evaluación:** precisión, recall, F1-score, matriz de confusión, validación cruzada estratificada (`StratifiedKFold`, `scoring="f1"`), curva ROC (AUC) y curva Precision-Recall (Average Precision).
9. **Selección justificada** del modelo final.

## Modelos entrenados

| Modelo | Configuración |
|---|---|
| Regresión Logística | `max_iter=1000`, `class_weight="balanced"` |
| Random Forest | `n_estimators=200`, `class_weight="balanced"` |

## Resultados

Evaluación sobre el conjunto de prueba (56.746 transacciones, 95 fraudes reales):

| Modelo | Accuracy | Precision | Recall | F1-score | AUC-ROC | Average Precision |
|---|---|---|---|---|---|---|
| Regresión Logística | 0,975 | 0,056 | 0,874 | 0,106 | 0,966 | 0,672 |
| **Random Forest** | **0,999** | **0,971** | 0,716 | **0,824** | 0,945 | **0,809** |

**Validación cruzada (5 folds, F1-score):**

| Modelo | F1 promedio | Desvío estándar |
|---|---|---|
| Regresión Logística | 0,114 | 0,013 |
| **Random Forest** | **0,837** | 0,046 |

> **Hallazgo clave:** el orden de los modelos se invierte según la métrica. La Regresión Logística tiene mayor AUC-ROC (0,966 vs. 0,945), pero Random Forest tiene mayor Average Precision (0,809 vs. 0,672). En datasets con un desbalance tan extremo, el AUC-ROC puede resultar engañosamente optimista, mientras que el Average Precision refleja con más fidelidad el costo real de cada modelo en falsas alarmas.

## Modelo final: Random Forest

Random Forest fue seleccionado como modelo final. Sobre el conjunto de prueba, detecta **68 de los 95 fraudes reales (71,6% de recall)** con una precisión del **97,1%**, generando apenas **2 falsos positivos sobre 56.651 transacciones legítimas**.

La Regresión Logística detecta una proporción mayor de fraudes (87,4% de recall), pero a costa de 1.393 falsos positivos —más de 16 falsas alarmas por cada fraude real encontrado—, un volumen inviable de operar en un equipo de revisión de fraude real.

## Limitaciones

- Las variables `V1`–`V28` están anonimizadas mediante PCA, lo que limita la interpretabilidad y el análisis de causas.
- Solo 492 fraudes en todo el dataset: cualquier evaluación puede ser sensible a cómo se distribuyen esos pocos casos entre folds o entre train/test.
- El modelo final deja 27 fraudes sin detectar sobre el conjunto de prueba.
- `class_weight="balanced"` fue la única estrategia de balanceo utilizada (no se probaron SMOTE, submuestreo ni ajuste de umbral).
- No se realizó búsqueda de hiperparámetros (`GridSearchCV` / `RandomizedSearchCV`).
- El dataset corresponde a solo dos días de septiembre de 2013 de tarjetahabientes europeos; un modelo en producción necesitaría reentrenarse periódicamente.

## Posibles mejoras futuras

- Probar SMOTE o undersampling para reducir los falsos negativos sin perder tanta precisión.
- Ajustar el umbral de decisión según el costo relativo de un falso positivo frente a un falso negativo.
- Optimizar hiperparámetros con `GridSearchCV` o `RandomizedSearchCV`.
- Probar modelos de gradient boosting (XGBoost, LightGBM, CatBoost).
- Analizar `feature_importances_` de Random Forest para entender qué componentes PCA aportan más.

## Tecnologías utilizadas

- Python 3.11
- pandas, numpy
- scikit-learn (`LogisticRegression`, `RandomForestClassifier`, `StratifiedKFold`, `class_weight`, `precision_recall_curve`, `average_precision_score`)
- matplotlib, seaborn
- kagglehub
- Google Colab

## Documentación y resultados

- **Reporte de resultados (PDF):** [Detección de Fraude – Tarjeta de Crédito](https://github.com/Juanjerezz/ml-talento-tech/blob/main/entrega%20final/Deteccion_Fraude_Tarjeta_Credito_Juan_Jerez_.pdf)
- **Notebook completo:** [Cuaderno_Proyecto_Final_credit_card_fraud.ipynb](https://github.com/Juanjerezz/ml-talento-tech/blob/main/entrega%20final/Cuaderno_Proyecto_Final_credit_card_fraud.ipynb)

## Cómo ejecutar el proyecto

1. Cloná el repositorio y abrí el notebook en Google Colab o Jupyter.
2. Instalá las dependencias:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn kagglehub
   ```
3. Descargá el dataset desde Kaggle (`mlg-ulb/creditcardfraud`) usando `kagglehub` o manualmente.
4. Ejecutá el notebook de principio a fin.

## Autor

**Juan Jerez**
Proyecto final — Curso de Machine Learning, Talento Tech

## Referencias

- Dataset: Machine Learning Group – ULB, *Credit Card Fraud Detection*, Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Documentación de scikit-learn: https://scikit-learn.org/stable/
