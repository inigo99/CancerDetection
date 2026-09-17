# Aprendizaje Multitarea para Detección de Cáncer en Mamografías

**Trabajo de Fin de Máster (TFM)** — Máster en Ciencia de Datos y Aprendizaje Automático, Escuela de Máster y Doctorado, curso académico 2022/2023.

Autor: **Íñigo Fernández Barrill** · Tutores: Manuel García Domínguez, Adrián Inés Armas

🇬🇧 *Looking for the English version? See [`README.md`](README.md).*

---

## Resumen

El cáncer de mama es una gran preocupación a nivel global, y la detección temprana juega un papel crucial en el resultado final. La mamografía es la herramienta más utilizada para detectarlo, ya que ofrece información útil en etapas tempranas de la enfermedad. Este trabajo estudia si una arquitectura de **Aprendizaje Multi Tarea (MTL)** — un único modelo entrenado para predecir varias variables relacionadas a la vez, en lugar de un modelo por variable — mejora la detección de cáncer frente a entrenar modelos individuales por separado.

Se exploran dos formas de combinar tareas:
1. **Fusión de características**: extraer un vector de características de la mamografía y combinarlo con datos del paciente (edad, densidad de la mama) para alimentar un segundo modelo, esta vez tabular, que predice el cáncer.
2. **Predicción conjunta**: un único modelo (CNN/Transformer) recibe la mamografía como entrada y predice a la vez el diagnóstico de cáncer, la edad del paciente y la densidad de la mama, compartiendo representaciones internas entre las tres tareas.

## Contenidos

- [Conjunto de datos](#conjunto-de-datos)
- [Resumen del proceso](#resumen-del-proceso)
- [Modelos y librerías](#modelos-y-librerías)
- [Resultados](#resultados)
  - [1. Modelos de imagen, conjunto desbalanceado](#1-modelos-de-imagen-conjunto-desbalanceado)
  - [2. Modelos de imagen, conjunto balanceado](#2-modelos-de-imagen-conjunto-balanceado)
  - [3. Modelos sobre datos tabulares](#3-modelos-sobre-datos-tabulares)
  - [4. Multi Tarea — fusión de características](#4-multi-tarea--fusión-de-características)
  - [5. Multi Tarea — predicción conjunta](#5-multi-tarea--predicción-conjunta)
  - [Resumen de resultados](#resumen-de-resultados)
- [Contenido del repositorio](#contenido-del-repositorio)
- [Cómo ejecutarlo](#cómo-ejecutarlo)

## Conjunto de datos

[RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) (Kaggle), con **54.706** imágenes de mamografías en formato DICOM y datos estructurados por imagen (identificador de paciente, lateralidad, vista, edad, etiqueta de cáncer, biopsia, invasivo, BIRADS, implante, densidad de la mama, identificador de la máquina).

**Reducción de volumen.** El conjunto original pesa 314,72 GB, inviable para trabajar directamente. Convertir DICOM → JPG (descartando los metadatos DICOM, ya que el nombre de archivo basta para localizar cada imagen en el CSV) reduce el tamaño a un tercio; redimensionar las imágenes desde hasta 4915×5355 px a 512×640 px deja el conjunto final en **5,72 GB**.

**Limpieza.** La variable `density` (densidad de la mama, muy predictiva) tiene muchos valores faltantes. En lugar de imputar con KNN (lo que introduciría datos sintéticos en un conjunto médico sensible), se descartan las filas con `density` o `age` faltante, pasando de 57.710 a **29.443** instancias. También se eliminan `site_id` e `implant` (un único hospital en los datos limpios; `implant` no es fiable en ese hospital). Distribución resultante de `density`: A 11 % · B 43 % · C 41 % · D 5 %.

**Desbalanceo de clases.** Sólo ~2 % de las imágenes están etiquetadas como positivas en cáncer. Se comparan dos estrategias de undersampling en lugar de oversampling/SMOTE, para no introducir datos sintéticos en una tarea diagnóstica donde los errores tienen un coste alto:
- *Por paciente*: 504 pacientes (252 positivos), 2.597 imágenes — sigue desbalanceado (25,6 % / 74,4 %) porque los pacientes positivos conservan tanto sus imágenes positivas como negativas.
- *Por imagen*: 1.328 imágenes, prácticamente balanceado al 50 % (664 positivas / 664 negativas) — el conjunto usado en el resto del trabajo.

**Particiones.** 80/20 entrenamiento/validación para los modelos de imagen; 70/10/20 entrenamiento/validación/prueba para los modelos tabulares (la validación se usa para elegir el mejor modelo e hiperparámetros, y después se reentrena con entrenamiento+validación y se evalúa una única vez sobre el conjunto de prueba).

## Resumen del proceso

```
DICOM (314,72 GB)
   │  → conversión a JPG + redimensionado a 512×640
   ▼
Conjunto limpio (29.443 filas, 5,72 GB)
   │  → se descartan density/age faltantes, site_id, implant
   ▼
Subconjunto balanceado por imagen (1.328 imágenes, ~50/50)
   │
   ├── Modelos de imagen (CNN/Transformer) ──────────────┐
   ├── Modelos tabulares sobre metadatos (edad, densidad…)├── Aprendizaje Multi Tarea
   └── Fusión imagen + metadatos / predicción conjunta ───┘
```

## Modelos y librerías

- **[FastAI](https://www.fast.ai/)** — bucle de entrenamiento de alto nivel (`DataBlock`, `Learner`, `fine_tune`, `EarlyStoppingCallback` sobre Valor-F, `SaveModelCallback`).
- **[Timm](https://github.com/huggingface/pytorch-image-models)** — arquitecturas preentrenadas: EfficientNet-B3, HRNet-W44, ResNet-50, ConvNeXt, DeiT3 (base, patch16, 384px).
- **PyTorch**, **scikit-learn** (`GridSearchCV` sobre Árbol C4.5/CART, Random Forest, Regresión Logística, MLP, SVM), **Pandas**, **NumPy**.
- Entrenado en Google Colab (CPU Intel Xeon, 13 GB RAM, GPU Tesla K80, 12 GB VRAM).

## Resultados

Todas las métricas son Valor-F (F1) sobre el conjunto de validación salvo que se indique lo contrario; cuanto más alto, mejor. Las cifras están tomadas directamente de las tablas de resultados de la memoria (`TFM.pdf`) y del notebook (`TFM.ipynb`).

### 1. Modelos de imagen, conjunto desbalanceado

Todas las arquitecturas (EfficientNet, HRNet, ResNet-50, ConvNeXt, DeiT3) alcanzan ~97,6–97,9 % de exactitud pero **Valor-F = 0,000**: con sólo ~2 % de positivos, los modelos simplemente predicen "sin cáncer" para todo. Es el fallo clásico de usar la exactitud como métrica en un conjunto médico desbalanceado, y la razón por la que fue necesario balancear los datos.

### 2. Modelos de imagen, conjunto balanceado

| Modelo | Exactitud | Precisión | Recuperación | **Valor-F** |
|---|---|---|---|---|
| EfficientNet-B3 | 0,5038 | 0,5556 | 0,2878 | 0,3792 |
| HRNet-W44 | 0,5606 | 0,5733 | 0,6475 | 0,6081 |
| **ResNet-50** | 0,5455 | 0,5422 | 0,8777 | **0,6703** |
| ConvNeXt | 0,5985 | 0,7324 | 0,3741 | 0,4952 |
| DeiT3 | 0,6439 | 0,6619 | 0,6619 | 0,6619 |

Sólo con balancear los datos, el mejor Valor-F pasa de 0,000 a **0,6703** (ResNet-50), y el modelo empieza a predecir realmente la clase positiva en lugar de colapsar en la clase mayoritaria.

### 3. Modelos sobre datos tabulares

Usando únicamente metadatos estructurados (lateralidad, vista, edad, densidad, máquina — sin imágenes). Tras la selección de características (probando todas las combinaciones de las cuatro variables clínicas), el mejor subconjunto fue **edad + densidad, sin normalizar**, evaluado con un SVM.

| Partición | Modelo | **Valor-F** |
|---|---|---|
| Validación, mejor subconjunto de variables | SVM | 0,6912 (Random Forest) / se elige SVM por generalización |
| **Conjunto de prueba, modelo final** | **SVM** | **0,6742** |

El pipeline tabular final (SVM sobre `age` + `density`) alcanza **Valor-F = 0,6742** en el conjunto de prueba — a la par del mejor modelo de imagen individual, usando sólo dos variables clínicas y ninguna imagen.

### 4. Multi Tarea — fusión de características

Un modelo de imagen (EfficientNet / HRNet / ResNet / ConvNeXt / DeiT) se trunca antes de sus últimas capas para producir un vector de 512 características por mamografía; ese vector se concatena con `age` y `density` y se pasa a los mismos modelos tabulares de la sección anterior.

| Extractor de características | Mejor modelo posterior | **Valor-F** |
|---|---|---|
| EfficientNet | SVM | 0,6704 |
| **HRNet** | SVM | **0,6736** |
| ResNet-50 | Random Forest / SVM | 0,6394 |
| ConvNeXt | Random Forest | 0,6569 |
| DeiT3 | SVM | 0,6385 |

Combinar características de imagen con metadatos mejora ligeramente el resultado tabular de referencia (mejor resultado Valor-F = 0,6736 con características de HRNet), aunque no de forma drástica; normalizar el conjunto combinado sacrifica el mejor resultado individual a cambio de más consistencia entre modelos.

### 5. Multi Tarea — predicción conjunta

Un único modelo predice `cancer`, `age` y `density` simultáneamente a partir de la mamografía, siguiendo una arquitectura inspirada en el [tutorial de FastAI multitarea de T. Dantas en Medium](https://medium.com/), con una función de pérdida combinada (MSE para la edad, entropía cruzada para densidad y cáncer) y ponderación por incertidumbre de cada tarea.

| Ponderación de la pérdida | Modelo | Valor-F (cáncer) | RMSE (edad) | Exactitud (densidad) |
|---|---|---|---|---|
| Pesos iguales | EfficientNet | 0,5733 | 0,0471 | 0,4886 |
| Pesos iguales | **ConvNeXt** | **0,6289** | 0,0650 | 0,0644 |
| Ponderada (0,6 cáncer / 0,3 densidad / 0,1 edad) | EfficientNet | 0,4228 | 0,4078 | 0,1629 |
| Ponderada (0,6 / 0,3 / 0,1) | ResNet-50 | 0,6250 | 0,0419 | 0,5492 |
| Ponderada (0,6 / 0,3 / 0,1) | **ConvNeXt** | **0,6305** | 0,0474 | 0,5492 |

Reponderar la función de pérdida hacia la tarea de cáncer (la que clínicamente importa) mejora los resultados en todas las arquitecturas salvo EfficientNet, con ConvNeXt alcanzando el mejor Valor-F en predicción conjunta: **0,6305**. Una prueba usando únicamente la pérdida de cáncer (sin supervisión de edad/densidad) da peores resultados que la versión multitarea equilibrada, confirmando que las tareas auxiliares sí aportan y no son sólo coste computacional extra.

### Resumen de resultados

| Enfoque | Mejor Valor-F (cáncer) |
|---|---|
| Sólo imagen, datos desbalanceados | 0,000 |
| Sólo imagen, datos balanceados (ResNet-50) | 0,6703 |
| Sólo datos tabulares (SVM, edad+densidad), conjunto de prueba | 0,6742 |
| Multi Tarea, fusión de características (HRNet + SVM) | 0,6736 |
| Multi Tarea, predicción conjunta (ConvNeXt, pérdida ponderada) | 0,6305 |

El balanceo de clases es, con diferencia, el factor con mayor impacto en este conjunto de datos: es lo que convierte un clasificador inútil (Valor-F = 0) en uno funcional. Los enfoques Multi Tarea se sitúan en el mismo rango de Valor-F (~0,63–0,67) que los mejores modelos individuales, sin superarlos claramente, y en este trabajo la fusión de características obtiene mejor resultado que la predicción conjunta. El apartado de conclusiones de la memoria quedó sin redactar en el PDF original, así que este resumen refleja únicamente los resultados reportados en cada sección de experimentos, y no es una transcripción de unas conclusiones formales del documento.

## Contenido del repositorio

- **`TFM.ipynb`** — notebook completo de experimentos: limpieza de datos, balanceo, todos los modelos de imagen/tabulares/multitarea y su evaluación (ver el badge de Colab al inicio del notebook).
- **`TFM.pdf`** — la memoria completa del TFM (en español), con el marco teórico, la metodología detallada y las tablas de resultados completas referenciadas arriba.

## Cómo ejecutarlo

1. Descarga el conjunto [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) de Kaggle.
2. Abre `TFM.ipynb` en Google Colab (o en un entorno local con `fastai`, `timm`, `torch`, `scikit-learn`, `pandas`, `numpy` instalados) y ejecuta primero los pasos de conversión DICOM→JPG y redimensionado descritos arriba.
3. Ejecuta las secciones del notebook en orden: limpieza de datos → balanceo → modelos de imagen → modelos tabulares → modelos multitarea.

---

*Este README se ha escrito a partir del contenido real de `TFM.ipynb` y `TFM.pdf` — todas las cifras anteriores proceden directamente de las tablas de resultados de la memoria.*
