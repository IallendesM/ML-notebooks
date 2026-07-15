# ML-Notebooks — INF398, USM (2023-2)

Talleres del curso **Introducción al Aprendizaje Automático (INF398)**, Universidad Técnica Federico Santa María. Desarrollados en pareja bajo el nombre de equipo **MCMC Metropolis**.

**Equipo:** Ignacio Allendes · Felipe Asbún

---

## 🏆 Resultados destacados

| Taller | Tema | Resultado |
|---|---|---|
| Taller 2 | Predicción de riesgo crediticio | 🥇 **1er lugar** — Kaggle privado (11 equipos) |
| Taller 4 | Predicción de precio de autos usados (domain adaptation) | 🥈 **2do lugar** — Kaggle privado (11 equipos) |

---

## Contenido

### Taller 1 — Clasificación de género musical
Dataset *Spotify Tracks DB* (~232.000 canciones). Pipeline completo de ML clásico: exploración y limpieza de datos, ingeniería de atributos, y clasificación multiclase de géneros musicales.
- Perceptrón implementado desde cero y con scikit-learn
- Regresión logística con estrategias **One-vs-Rest** y **One-vs-One**
- Validación cruzada, matriz de confusión, importancia de atributos (permutation importance)

### Taller 2 — Predicción de riesgo crediticio 🥇
Predicción de default de clientes a partir de múltiples fuentes de datos (historial de préstamos, movimientos de tarjeta, solicitudes).
- Feature engineering sobre datos multi-tabla (joins y agregaciones por cliente)
- Pipeline con `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`
- Selección de componentes vía PCA con validación cruzada anidada (`StratifiedKFold`)
- Modelo final: Regresión Logística con balanceo de clases, optimizado para AUC

### Taller 3 — SVM y Árboles de Decisión
Predicción de nivel de ingreso (clasificación binaria) sobre datos tabulares tipo censo.
- SVM lineal y con kernel RBF; selección de kernel e hiperparámetros
- Árboles de decisión con distintas estrategias de regularización (profundidad, poda)
- Comparación de curvas de aprendizaje con datasets de distinto tamaño

### Taller 4 — Predicción de precio de autos usados (Domain Adaptation) 🥈
Predicción de precios con **shift de dominio** entre datos de origen y de destino (distintas zonas geográficas).
- Análisis de distribuciones con KDE plots para detectar el shift de dominio
- Selección de atributos vía información mutua (`mutual_info_regression`)
- Dos enfoques comparados: balanceo con SMOTEENN + Gradient Boosting, vs. stacking de modelos entrenados por separado en dominio fuente y objetivo

---

## Stack técnico
`Python` · `pandas` · `numpy` · `scikit-learn` · `seaborn` / `matplotlib` · `imbalanced-learn`

---

## Nota
Estos notebooks fueron desarrollados como entregas del curso y están orientados a los enunciados originales de cada taller. Los Talleres 1 y 3 fueron ejercicios guiados de aprendizaje; mientras que los Talleres 2 y 4 fueron evaluados en competencias de Kaggle entre 11 equipos del curso.
