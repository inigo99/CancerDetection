# Multi-Task Learning for Breast Cancer Detection in Mammographies

**Master's Thesis (TFM)** — Master's Degree in Data Science and Machine Learning, Escuela de Máster y Doctorado, academic year 2022/2023.

Author: **Íñigo Fernández Barrill** · Supervisors: Manuel García Domínguez, Adrián Inés Armas

🇪🇸 *¿Buscas la versión en español? Está en [`README.es.md`](README.es.md).*

---

## Abstract

Breast cancer is a major global health concern, and early detection plays a crucial role in patient outcomes. Mammography is the most widely used screening tool, offering useful information at early stages of the disease. This project studies whether a **Multi-Task Learning (MTL)** architecture — a single model trained to predict several related targets at once, rather than one model per target — improves cancer detection over training individual models separately.

Two ways of combining tasks are explored:
1. **Feature-level fusion**: extract an image embedding from a mammography and combine it with patient metadata (age, breast density) to feed a second, tabular model that predicts cancer.
2. **Joint prediction**: a single CNN/Transformer backbone takes the mammography as input and predicts cancer diagnosis, patient age and breast density simultaneously, sharing internal representations across the three tasks.

## Contents

- [Dataset](#dataset)
- [Pipeline overview](#pipeline-overview)
- [Models and libraries](#models-and-libraries)
- [Results](#results)
  - [1. Image-only baseline (imbalanced data)](#1-image-only-baseline-imbalanced-data)
  - [2. Image-only, balanced dataset](#2-image-only-balanced-dataset)
  - [3. Tabular metadata models](#3-tabular-metadata-models)
  - [4. Multi-task — feature fusion](#4-multi-task--feature-fusion)
  - [5. Multi-task — joint prediction](#5-multi-task--joint-prediction)
  - [Conclusions](#conclusions)
- [Repository contents](#repository-contents)
- [How to run it](#how-to-run-it)

## Dataset

[RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) (Kaggle), consisting of **54,706** mammography images in DICOM format plus structured per-image metadata (patient ID, laterality, view, age, cancer label, biopsy, invasive, BIRADS, implant, breast density, imaging machine ID).

**Volume reduction.** The raw dataset is 314.72 GB, which is impractical to work with directly. Converting DICOM → JPG (dropping DICOM metadata not needed here, since the filename already maps to the CSV) reduces it to a third of its size; resizing images from up to 4915×5355 px down to 512×640 px brings the final working set to **5.72 GB**.

**Cleaning.** The `density` variable (breast density, a strong predictor) has many missing values. Rather than imputing with KNN (which would introduce synthetic values into a sensitive medical dataset), rows with missing `density` or `age` were dropped, taking the dataset from 57,710 to **29,443** instances. `site_id` and `implant` were dropped too (single hospital in the cleaned data; `implant` not reliably recorded there). Resulting `density` distribution: A 11% · B 43% · C 41% · D 5%.

**Class imbalance.** Only ~2% of images are labelled positive for cancer. Two undersampling strategies were compared instead of oversampling/SMOTE, to avoid introducing synthetic data into a diagnostic task where errors carry a high cost:
- *By patient*: 504 patients (252 positive), 2,597 images — still imbalanced (25.6% / 74.4%) because positive patients keep both their positive and negative images.
- *By image*: 1,320 images, an almost perfectly balanced 664 positive / 656 negative — the set used for all subsequent modelling (80% train / 20% validation for image models: 1,056 / 264 images).

**Splits.** 80/20 train/validation for image models; 70/10/20 train/validation/test for tabular models (validation used to pick the best model/hyperparameters, then retrained on train+validation and evaluated once on the held-out test set).

## Pipeline overview

```
DICOM (314.72 GB)
   │  → JPG conversion + resize to 512×640
   ▼
Cleaned dataset (29,443 rows, 5.72 GB)
   │  → drop missing density/age, drop site_id/implant
   ▼
Balanced-by-image subset (1,320 images, ~50/50)
   │
   ├── Image-only CNN/Transformer models  ───────────────┐
   ├── Tabular models on metadata (age, density, ...)    ├── Multi-Task Learning
   └── Image + metadata fusion / joint prediction ───────┘
```

## Models and libraries

- **[FastAI](https://www.fast.ai/)** — high-level training loop (`DataBlock`, `Learner`, `fine_tune`, `EarlyStoppingCallback` on F1, `SaveModelCallback`).
- **[Timm](https://github.com/huggingface/pytorch-image-models)** — pretrained backbones: EfficientNet-B3, HRNet-W44, ResNet-50, ConvNeXt, DeiT3 (base, patch16, 384px).
- **PyTorch**, **scikit-learn** (`GridSearchCV` over Decision Tree C4.5/CART, Random Forest, Logistic Regression, MLP, SVM), **Pandas**, **NumPy**.
- Trained on Google Colab (Intel Xeon CPU, 13 GB RAM, Tesla K80 GPU, 12 GB VRAM).

## Results

All metrics below are F1-score ("Valor-F") on the validation set unless noted; higher is better. Numbers are taken directly from the thesis document (`TFM.pdf`) and the accompanying notebook (`TFM.ipynb`).

### 1. Image-only baseline (imbalanced data)

Every architecture (EfficientNet, HRNet, ResNet-50, ConvNeXt, DeiT3) reaches ~97.6–97.9% accuracy but **F1 = 0.000**: with only ~2% positives, the models simply predict "no cancer" for everything. This is the textbook failure mode of accuracy on an imbalanced medical dataset, and the reason balancing was necessary.

### 2. Image-only, balanced dataset

| Model | Accuracy | Precision | Recall | **F1** |
|---|---|---|---|---|
| EfficientNet-B3 | 0.5038 | 0.5556 | 0.2878 | 0.3792 |
| HRNet-W44 | 0.5606 | 0.5733 | 0.6475 | 0.6081 |
| **ResNet-50** | 0.5455 | 0.5422 | 0.8777 | **0.6703** |
| ConvNeXt | 0.5985 | 0.7324 | 0.3741 | 0.4952 |
| DeiT3 | 0.6439 | 0.6619 | 0.6619 | 0.6619 |

Balancing alone takes the best F1 from 0.000 to **0.6703** (ResNet-50), and the model now genuinely predicts the positive class instead of collapsing to the majority class.

### 3. Tabular metadata models

Using only structured metadata (laterality, view, age, density, machine ID — no images). After feature selection (testing all subsets of the four clinical variables), the best-performing subset was **age + density alone, unnormalised**, evaluated with an SVM.

| Split | Model | **F1** |
|---|---|---|
| Validation, best feature subset | SVM | 0.6912 (Random Forest) / SVM selected for generalisation |
| **Test set, final model** | **SVM** | **0.6742** |

The final tabular pipeline (SVM on `age` + `density`) reaches **F1 = 0.6742** on the held-out test set — on par with the best single-task image model, using only two clinical variables and no imaging at all.

### 4. Multi-task — feature fusion

Three ways of combining image and tabular data were considered — **Early Fusion** (merge raw inputs before feature extraction), **Middle Fusion** (extract image features first, then merge with tabular data before classifying) and **Late Fusion** (train separate models and merge their outputs). **Middle Fusion** was chosen as the best fit: an image backbone (EfficientNet / HRNet / ResNet / ConvNeXt / DeiT) is truncated before its final layers to produce a 512-dimensional embedding per mammography, which is then concatenated with `age` and `density` and fed into the same tabular models as above.

| Feature extractor | Best downstream model | **F1** |
|---|---|---|
| EfficientNet | SVM | 0.6704 |
| **HRNet** | **Random Forest** | **0.6788** |
| ResNet-50 | Random Forest / SVM | 0.6394 |
| ConvNeXt | Random Forest | 0.6569 |
| DeiT3 | SVM | 0.6385 |

Combining image features with metadata gives the best single result in the whole study: **F1 = 0.6788** (HRNet features + Random Forest), edging out both the tabular-only and image-only baselines — though, as the thesis's conclusions note, not by a wide enough margin to call it a clear win for Multi-Task Learning (see below).

### 5. Multi-task — joint prediction

A single backbone predicts `cancer`, `age` and `density` simultaneously from the mammography alone, following an architecture inspired by [T. Dantas' FastAI multi-task tutorial](https://medium.com/), with a combined loss (MSE for age, cross-entropy for density and cancer) and per-task uncertainty weighting.

| Loss weighting | Model | F1 (cancer) | RMSE (age) | Accuracy (density) |
|---|---|---|---|---|
| Equal weights | EfficientNet | 0.5733 | 0.0471 | 0.4886 |
| Equal weights | **ConvNeXt** | **0.6289** | 0.0650 | 0.0644 |
| Weighted (0.6 cancer / 0.3 density / 0.1 age) | EfficientNet | 0.4228 | 0.4078 | 0.1629 |
| Weighted (0.6 / 0.3 / 0.1) | ResNet-50 | 0.6250 | 0.0419 | 0.5492 |
| Weighted (0.6 / 0.3 / 0.1) | **ConvNeXt** | **0.6305** | 0.0474 | 0.5492 |

Re-weighting the combined loss towards the cancer task (which is what actually matters clinically) improves results for every backbone except EfficientNet, with ConvNeXt reaching the best joint-prediction F1 of **0.6305**. An ablation using only the cancer loss term (dropping age/density supervision) performs worse than the balanced multi-task version — confirming that the auxiliary tasks do help, they aren't just extra compute.

### Conclusions

| Approach | Best F1 (cancer) |
|---|---|
| Image only, imbalanced data | 0.000 |
| Image only, balanced data (ResNet-50) | 0.6703 |
| Tabular metadata only (SVM, age+density), test set | 0.6742 |
| **Multi-task, feature fusion (HRNet + Random Forest)** | **0.6788** |
| Multi-task, joint prediction (ConvNeXt, weighted loss) | 0.6305 |

These are the thesis's own conclusions (Section 6):

- **Data treatment matters most.** The dataset needed heavy volume reduction and cleaning before it was usable — the final working set is only about **2.5%** of the original data volume. How the data is processed has a direct, large effect on the final results.
- **Multi-Task Learning did not bring a clear improvement.** Ranked by F1, the four approaches land close together (0.6788 → 0.6742 → 0.6703 → 0.6305), so there's no strong evidence that combining tasks beats training a single well-tuned model on this particular problem.
- **Multi-Task Learning is expensive.** Both MTL approaches are slower to train than a single-task model, and feature fusion (Approach 1) is by far the heaviest: it requires training an image model, extracting features, fusing the data, and then training a second model on top.
- **Future work:** an ensemble of the separate image and tabular models could plausibly reach similar results at a lower time cost than either Multi-Task architecture — a cheaper alternative worth exploring.

## Repository contents

- **`TFM.ipynb`** — full experiment notebook: data cleaning, balancing, all image/tabular/multi-task models and their evaluation ([open in Colab](https://colab.research.google.com/) — see the badge at the top of the notebook).
- **`TFM.pdf`** — the full thesis write-up (in Spanish), with the theoretical background, detailed methodology, and complete results tables referenced above.

## How to run it

1. Get the [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) dataset from Kaggle.
2. Open `TFM.ipynb` in Google Colab (or a local environment with `fastai`, `timm`, `torch`, `scikit-learn`, `pandas`, `numpy` installed) and run the DICOM→JPG conversion / resizing steps described above before training.
3. Run the notebook sections in order: data cleaning → balancing → image models → tabular models → multi-task models.

---

*This README was written from the actual contents of `TFM.ipynb` and `TFM.pdf` — all figures and conclusions above are taken directly from the thesis's results tables and its Conclusions section.*
