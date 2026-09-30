# Model card: breast cancer detection in mammographies (Master's thesis)

This card covers the models trained in `TFM.ipynb` for the thesis *Multi-Task Learning for Breast Cancer Detection in Mammographies* (University of La Rioja, 2022/2023). No trained weights are published.

## Intended use

**Research and teaching only.** These models compare Multi-Task Learning with single-task models on a small, balanced subset of a public dataset. They are **not** a medical device. Do not use them to screen, diagnose or triage real patients.

## Models

| Approach | Model | Inputs | Output |
|---|---|---|---|
| Image only | ResNet-50, EfficientNet-B3, HRNet-W44, ConvNeXt, DeiT3 (Timm, ImageNet-pretrained, FastAI fine-tuning) | Mammography, 512×640 JPG | Cancer yes/no |
| Tabular | SVM (also C4.5, CART, logistic regression, MLP, random forest) | `age`, `density` | Cancer yes/no |
| Multi-task, feature fusion | 512-d image embedding + `age` + `density` → random forest / SVM | Mammography + metadata | Cancer yes/no |
| Multi-task, joint prediction | One backbone with three heads, uncertainty-weighted loss (0.1 / 0.6 / 0.3) | Mammography | Age, cancer, density |

## Data

[RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) (Kaggle). Rows missing `age` or `density` were dropped, leaving 29,443 images. About 2% of them are positive, so the data was undersampled to 1,320 images (664 positive / 656 negative). All images come from a single site. Positives are therefore over-represented about 25 times compared with real screening.

## Metrics

F1 for the cancer class, on validation (or on test for the tabular model). The full tables are in the [README](README.md#results).

| Approach | Best F1 |
|---|---|
| Image only, imbalanced | 0.000 |
| Image only, balanced (ResNet-50) | 0.6703 |
| Tabular, age + density (SVM, test) | 0.6742 |
| Feature fusion (HRNet + random forest) | 0.6788 |
| Joint prediction (ConvNeXt, weighted loss) | 0.6305 |

## Known issues in the notebook

These issues were found while turning the notebook into the tested package in `src/`. The package fixes each one, and each fix has a test in `tests/test_core.py`. The thesis figures above come from the notebook as it was, so read them with these issues in mind.

1. **Patient leakage between train and validation.** The image models split by image, so two views of the same breast can land on both sides. Fix: `data.patient_split`.
2. **Negatives drawn with replacement.** `np.random.choice` defaults to `replace=True`, so some negatives were picked twice. That is why the balanced set is 664 / 656 instead of an exact 50/50. Fix: `data.balance_by_image`.
3. **A separate scaler on each split.** `StandardScaler` was fitted on train, validation and test independently. This only affects the normalised runs; the final SVM uses unnormalised features. Fix: a pipeline in `tabular.svm_baseline`.
4. **Feature fusion paired rows by position.** The training embeddings came from a shuffled dataloader. The metadata came from `df.iloc` on indices into the image-file list, not into the dataframe. Embeddings and labels therefore very likely do not line up, which would explain why fusion scores the same as age + density alone. Fix: `fusion.fusion_frame` joins on `image_id`.
5. **Flipped cancer label in the joint-prediction F1.** `f1_cancer` predicted class 1 when the logit of class 0 was larger. Early stopping monitored that metric. Fix: `multitask.f1_cancer`.

## Limitations

- Validation sets are small (264 images), so differences of about ±0.03 in F1 are within noise.
- The data comes from one site and one population, with no external validation.
- There is no calibration and no analysis by subgroup (age, density, machine).
- Balanced sampling means precision and F1 here do not reflect real-world screening prevalence.
