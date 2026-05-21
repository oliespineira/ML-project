# Film Revenue Prediction

**Machine Learning Foundations — IE University**
Prof. Matteo Turilli

A complete multimodal machine learning pipeline for **pre-release film revenue prediction** using structured metadata, synopsis embeddings, poster embeddings, and talent-history features.

The project investigates a central machine learning engineering question:

> *What matters more for pre-release prediction performance: richer feature representations or more expressive model families?*

The pipeline combines:

- TMDB financial and metadata records
- IMDb structured talent information
- MPNet synopsis embeddings
- CLIP poster embeddings
- Temporal leakage-safe talent-history features

and evaluates:

- Ridge Regression
- LightGBM
- XGBoost
- CatBoost
- Multiple ensemble strategies

---

## Final Results

| Model                     | Test RMSE ↓ | Test R² ↑ |
| ------------------------- | ----------: | --------: |
| Ridge baseline            |       2.167 |     0.358 |
| LightGBM v1               |       1.775 |     0.652 |
| LightGBM v2               |       1.749 |     0.662 |
| XGBoost                   |       1.718 |     0.675 |
| CatBoost                  |       1.738 |     0.668 |
| Weighted Average Ensemble |   **1.711** | **0.678** |
| Stacking Ensemble         |       1.712 |     0.677 |

**Best overall model:** Weighted Average Ensemble — Test RMSE: **1.711** | Test R²: **0.678**

---

## Key Findings

1. **Feature engineering contributed the largest gains.** Most performance improvement came from multimodal feature representations rather than from switching algorithms.

2. **Gradient boosting dramatically outperformed linear baselines.** Ridge regression explained ~36% of revenue variance; multimodal boosted models explained ~68%.

3. **Poster embeddings helped tree-based models more than linear models.** Visual features provided modest but measurable improvements for LightGBM.

4. **Ensembling provided limited gains.** Residual correlations between models remained very high (0.97–0.98), limiting ensemble diversity.

5. **A substantial irreducible noise floor remains.** Word-of-mouth, critical reception, release competition, and cultural timing are fundamentally unknowable before release.

---

## Repository Structure

```text
.
├── notebook.ipynb          # Main project notebook (full end-to-end ML pipeline)
├── outputs/                # Saved plots, embeddings, and figures
├── data/
│   └── raw/                # TMDB + IMDb datasets
├── README.md
└── requirements.txt        # Python dependencies
```

---

## Dataset Sources

### 1. TMDB Dataset (Kaggle)

Used for: budget, revenue, runtime, genres, release date, overview text, poster paths, and production metadata.

- `TMDB_movie_dataset_v11.csv`

### 2. IMDb Non-Commercial Datasets

Used for: directors, writers, top-billed cast, and talent-history features.

- `title.crew.tsv`
- `title.principals.tsv`
- `name.basics.tsv`

### 3. Poster Images

Downloaded dynamically from the TMDB CDN:

```text
https://image.tmdb.org/t/p/w342/{poster_path}
```

---

## How to Run the Project

### Step 1 — Clone the repository

```bash
git clone <repo-url>
cd <repo-folder>
```

### Step 2 — Create a virtual environment

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable, install manually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm xgboost catboost shap sentence-transformers transformers torch optuna
```

### Step 4 — Download and place the required data files

Create the following directory structure at the project root and place the downloaded datasets inside it:

```text
data/
└── raw/
    ├── TMDB_movie_dataset_v11.csv
    ├── title.crew.tsv
    ├── title.principals.tsv
    └── name.basics.tsv
```

- **TMDB dataset:** download `TMDB_movie_dataset_v11.csv` from [Kaggle](https://www.kaggle.com/)
- **IMDb datasets:** download the `.tsv` files from [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/)

### Step 5 — Launch the notebook

```bash
jupyter notebook
```

Open `notebook.ipynb` and run cells sequentially from top to bottom.

**Important notes before running:**

- Embedding generation (synopsis + poster) can take significant time on first run; embeddings are cached to disk automatically and loaded on subsequent runs.
- Poster downloads require an active internet connection.
- Later pipeline stages (XGBoost, CatBoost, ensembles, SHAP) require all earlier cells to have run successfully.

---

## Hardware and Runtime Notes

The notebook is designed for local execution on consumer hardware.

**Recommended specs:**

- 16–24 GB RAM
- Python 3.12+
- Apple Silicon or CUDA-capable GPU (optional but speeds up embedding generation)

The pipeline includes memory diagnostics, embedding caching, garbage collection, and checkpoint saving to reduce crashes during long-running embedding stages.

---

## Machine Learning Pipeline Overview

### 1. Data Acquisition

Loading TMDB and IMDb records, poster retrieval, and reproducible project paths.

### 2. Pre-Cleaning EDA

Missing-value analysis, duplicate detection, financial outlier inspection, and language/country distributions.

### 3. Data Merging

TMDB ↔ IMDb joins, director and cast mapping, and ID consistency validation.

### 4. Data Cleaning

Removing unreleased films and invalid financial records, deduplicating titles, enforcing structural completeness, and defining supervised targets.

### 5. Leakage Prevention

Explicit exclusion of `vote_count`, `vote_average`, and `popularity`. Temporal split ordered chronologically by release date: 70% train / 15% validation / 15% test.

### 6. Feature Engineering

**Structured features:** inflation-adjusted budget, runtime, release month/year, language indicators, country indicators, sequel flags.

**Genre features:** one-hot encoded genres.

**Synopsis embeddings:** `all-MiniLM-L6-v2`, upgraded to `all-mpnet-base-v2`.

**Poster embeddings:** CLIP visual embeddings with PCA dimensionality reduction.

**Talent-history features:** leakage-safe historical statistics (director median revenue, cast median revenue) computed strictly from past films only.

### 7. Modeling

Ridge Regression, LightGBM, XGBoost, CatBoost, Stacking Ensemble, and Weighted Average Ensemble.

### 8. Interpretability

SHAP importance analysis, beeswarm plots, residual analysis, and ablation studies.

---

## Notebook Sections

1. Project Overview
2. Data Acquisition
3. Pre-Cleaning EDA
4. Data Merging
5. Data Cleaning
6. Post-Cleaning EDA
7. Temporal Train / Val / Test Split
8. Leakage Analysis
9. Feature Engineering
10. Multimodal Fusion
11. Baseline Modeling
12. Evaluation Metrics
13. LightGBM Modeling
14. Ablation Study
15. SHAP Interpretability
16. Error Analysis
17. Final Test Evaluation
18. Pitfalls Checklist
19. Improved Pipeline (inflation-adjusted budget, poster PCA, MPNet embeddings, Optuna tuning, dedicated classifier)
20. Improved Final Evaluation
21. Baseline vs. Improved Comparison
22. Final Pitfalls Checklist
23. Ensemble Learning Motivation
24. Additional Dependencies
25. XGBoost
26. CatBoost
27. Residual Diversity Analysis
28. Stacking Ensemble
29. Weighted Average Ensemble
30. Ensemble Final Evaluation
31. Full Model Comparison
32. Poster Embedding Ablation
33. Final Conclusions

---

## Reproducibility

The project emphasizes strict reproducibility throughout:

- Fixed `SEED = 42` across all models
- Train-only fitting of scalers and PCA
- Chronological train/val/test split with no data leakage
- Cached embeddings for deterministic re-runs
- No test-set tuning at any stage

---

## Limitations

- Strong temporal distribution shift after 2017
- Limited franchise and IP metadata
- Cold-start problem for debut talent
- Imperfect pre-release information boundary
- Significant irreducible uncertainty inherent to film revenue forecasting

---

## Future Work

- Larger multimodal vision-language encoders
- Franchise and IP knowledge graphs
- Bayesian talent priors
- Streaming-era adjustment features
- Transformer-based tabular fusion
- Uncertainty calibration
- Hierarchical temporal models

---

## Authors

Developed as a full-course machine learning engineering project at **IE University** focused on multimodal learning, leakage prevention, feature engineering, reproducible ML workflows, model interpretability, ensemble learning, and scientific evaluation.
