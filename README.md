# Urban Heat Island Analysis — Rome

This repository contains the code, data, and outputs for a study on Urban Heat Island (UHI) intensity in Rome, Italy. The project predicts **Land Surface Temperature (LST)** at 30 m resolution using multi-source satellite imagery and a LightGBM gradient boosting model. The work follows a similar methodology to [Zhao et al., 2025 (*Scientific Reports*)](s41598-025-04025-0.pdf) but is applied to Rome with several methodological adaptations.

---

## Project Overview

The pipeline pulls three satellite data sources over the same 30 m UTM grid (EPSG:32633, summer 2025), engineers features, merges them, and trains a LightGBM regressor to predict LST. Two splitting strategies are compared: a random pixel split (standard interpolation) and a spatial block split (more realistic generalisation test). Scenario analyses then use the trained model to estimate the cooling effect of urban interventions.

---

## Repository Structure

```
.
├── Data/                                        # Input and processed datasets (not tracked in git — see below)
│   ├── rome_s1_features.csv                     # Sentinel-1 SAR features (677,735 pixels)
│   ├── rome_s2_features_v4.csv                  # Sentinel-2 optical features + SRTM elevation
│   ├── rome_final_landsat.csv                   # Landsat 8/9 LST target values (°C)
│   ├── rome_srtm_elevation_30m.csv              # SRTM elevation (30 m)
│   ├── rome_dataset_features_target_with_focal.csv  # Final merged dataset (34 features + LST)
│   ├── train_spatial_split.csv                  # Spatial block training set
│   ├── val_spatial_split.csv                    # Spatial block validation set
│   └── test_spatial_split.csv                   # Spatial block test set
│
├── Feature Maps/                                # Spatial visualisations of features and the AOI
│   ├── aoi_map.png
│   ├── sentinel1_map.png
│   ├── sentinel2_map.png
│   ├── lst_map.png
│   └── split_map.png
│
├── Shap Values/                                 # SHAP-based feature attribution plots
│   ├── global_feature_importance.png
│   ├── beeswarm.png
│   └── shap_dependence_plot.png
│
├── Scenario Analysis Graphs/                    # Predicted LST change under urban interventions
│   ├── v1_greening_*.png                        # Urban greening scenario
│   ├── v1_coolroofs_*.png                       # Cool roofs scenario
│   └── v1_densification_*.png                   # Urban densification scenario
│
├── sentinel1_features_rome.ipynb                # Step 1 — Sentinel-1 feature extraction
├── sentinel2_features_rome.ipynb                # Step 2 — Sentinel-2 feature extraction + SRTM
├── landsat_lst_rome.ipynb                       # Step 3 — Landsat LST collection
├── data_collection.ipynb                        # Step 4 — Merge, focal features, baseline model
├── lightgbm_5fold_cv.ipynb                      # Step 5a — 5-fold CV (random split)
├── lgbm_spatial_splitting.ipynb                 # Step 5b — Spatial split, final model, SHAP
│
├── tuned_lgbm_model_spatial.pkl                 # Serialised final trained model
├── sentinel1_feature_documentation.docx         # Description of all S1 features
└── s41598-025-04025-0.pdf                       # Reference paper (Zhao et al., 2025)
```

---

## Notebooks

### `sentinel1_features_rome.ipynb` — Sentinel-1 SAR Feature Extraction
Uses Google Earth Engine to collect **Sentinel-1 GRD** imagery over Rome (ascending orbit, summer 2025). Applies a refined Lee speckle filter, builds a summer median composite, and engineers 9 features:

| Feature | Description |
|---|---|
| `VV`, `VH` | Raw backscatter (dB) — surface roughness / vegetation volume |
| `VV_VH_diff` | Urban vs. vegetation discriminator |
| `VH_VV_ratio` | Cross-polarisation fraction — canopy depth proxy |
| `RFDI` | Radar Forest Degradation Index: +1 = urban/bare, −1 = forest |
| `SAR_urban` | Double-bounce index — building density proxy |
| `VV_std`, `VH_std` | Annual temporal std — construction activity / vegetation seasonality |
| `texture_contrast` | GLCM contrast of VV — urban canyon spatial heterogeneity |

Exports a GeoTIFF to Google Drive and reads it back as `rome_s1_features.csv`.

---

### `sentinel2_features_rome.ipynb` — Sentinel-2 Optical Feature Extraction + SRTM Elevation
Collects **Sentinel-2 SR Harmonised** imagery (pixel-level SCL cloud masking, 51 scenes). Computes 6 spectral/land-cover features and also exports SRTM elevation:

| Feature | Description |
|---|---|
| `NDVI` | Vegetation density → evapotranspiration / cooling |
| `NDBI` | Built-up intensity → heat absorption |
| `NDWI` | Surface water / moisture |
| `Albedo` | Solar reflectivity → net heat gain |
| `PIS` | Fractional impervious surface (0–1, continuous via `reduceResolution`) |
| `PGS` | Fractional green surface (0–1, continuous) |
| `elevation` | SRTM 30 m digital elevation model |

All outputs use an explicit `crsTransform` to align pixel-for-pixel with the Sentinel-1 and Landsat grids. Saves `rome_s2_features_v4.csv` and `rome_srtm_elevation_30m.csv`.

---

### `landsat_lst_rome.ipynb` — Landsat LST Collection
Builds a median summer composite from **Landsat 8 + 9 Collection 2 Level-2** (11 images, < 5% cloud cover). Converts the `ST_B10` thermal band to °C, exports a GeoTIFF using the same `crsTransform` as the other datasets, and saves `rome_final_landsat.csv`. LST range over Rome: **31.5 – 58.6 °C**.

---

### `data_collection.ipynb` — Data Merging, Focal Features, and Baseline Model
Merges the three per-source CSVs on `x_utm` / `y_utm` (677,735 pixels, 22 columns). Then engineers an additional 17 **focal (neighbourhood) features** for each original feature using a 5 × 5 pixel (150 m) spatial mean filter, doubling the feature set to 34 predictors. Saves the final dataset as `rome_dataset_features_target_with_focal.csv`.

Also contains exploratory LightGBM baseline models trained on a random 60/20/20 split, showing the improvement from adding focal features (R² rising from 0.73 to 0.89 on the test set).

---

### `lightgbm_5fold_cv.ipynb` — 5-Fold Cross-Validation (Random Split)
Runs shuffled **5-fold cross-validation** on the full 677k-pixel dataset using tuned hyperparameters. Reports per-fold and summary metrics and saves feature importance rankings. Results:

- Validation R²: **0.756 ± 0.001**
- Validation MAE: **1.13 ± 0.002 °C**
- Train–validation R² gap: **0.002** (no meaningful overfitting)

> Note: these metrics measure within-Rome interpolation, not spatial generalisation.

---

### `lgbm_spatial_splitting.ipynb` — Spatial Block Split, Final Model, and Scenario Analysis
Implements a **3 km spatial block split** (70/15/15 train/val/test) using `GroupKFold` on block IDs to prevent spatial autocorrelation leakage. Trains the final model, evaluates on geographically held-out pixels, and computes SHAP values to interpret feature contributions. Also contains the **scenario analysis**: simulates predicted LST changes under three urban interventions (urban greening, cool roofs, densification) and saves outputs to `Scenario Analysis Graphs/`. The final tuned model is serialised as `tuned_lgbm_model_spatial.pkl`.

---

## Feature Set Summary

| Source | Features | Count |
|---|---|---|
| Sentinel-1 | VV, VH, VV_VH_diff, VH_VV_ratio, RFDI, SAR_urban, VV_std, VH_std, texture_contrast | 9 |
| Sentinel-2 | NDVI, NDBI, NDWI, Albedo, PIS, PGS, NLI | 7 |
| Topography | elevation | 1 |
| Focal (5×5 neighbourhood mean of all above) | *_focal variants | 17 |
| **Total** | | **34** |

---

## Dependencies

```
earthengine-api
geemap
rasterio
pyproj
lightgbm
scikit-learn
pandas
numpy
scipy
matplotlib
seaborn
shap
```

A free [Google Earth Engine](https://earthengine.google.com) account and a GCP project ID are required to re-run the data collection notebooks.

### Reproducing the Data

The `Data/` folder is **not tracked in this repository** because the CSV files are too large for git. To regenerate the data from scratch, run the notebooks in order:

1. `sentinel1_features_rome.ipynb` → exports `rome_s1_features.csv`
2. `sentinel2_features_rome.ipynb` → exports `rome_s2_features_v4.csv` and `rome_srtm_elevation_30m.csv`
3. `landsat_lst_rome.ipynb` → exports `rome_final_landsat.csv`
4. `data_collection.ipynb` → merges all three and produces `rome_dataset_features_target_with_focal.csv` plus the spatial split files

Steps 1–3 require GEE authentication and export GeoTIFFs to Google Drive, which must then be downloaded locally before running the next step.

---

## Reference

> Zhao et al. (2025). *[Title]*. *Scientific Reports*, 15. [s41598-025-04025-0.pdf](s41598-025-04025-0.pdf)
