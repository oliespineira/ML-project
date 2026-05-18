# Film Revenue Prediction

A multimodal machine learning project that predicts pre-release film box-office revenue using structured metadata, text embeddings, poster image embeddings, and talent track-records.

**Primary artifact:** [`notebook.ipynb`](notebook.ipynb) — a single self-contained Jupyter notebook covering the full ML pipeline across 33 numbered sections.

---

## Research Question

> What matters more for pre-release revenue prediction: feature representation or model choice?

**Primary target:** `y = log1p(revenue)` — log-transforming revenue compresses the heavy right tail (thousands to billions of dollars) so every film contributes meaningfully to training rather than being dominated by blockbusters.

**Derived classification target:** `profitable = 1[revenue > 1.5 × budget]` — evaluates whether the model can identify profitable films as a secondary task.

---

## Repository Structure

```
ml-movies-model/
├── notebook.ipynb              # full pipeline (198 cells, 33 sections)
├── requirements.txt            # python dependencies
├── ml_project.pdf              # project report
├── data/
│   ├── raw/                    # raw input files (not committed — see Data Sources)
│   │   ├── TMDB_movie_dataset_v11.csv
│   │   ├── title.crew.tsv.csv
│   │   ├── title.principals.tsv.csv
│   │   └── name.basics.tsv.csv
│   └── processed/              # pre-computed embedding caches (committed to repo)
│       ├── synopsis_embeddings.npy / synopsis_ids.npy
│       ├── synopsis_embeddings_mpnet.npy / synopsis_ids_mpnet.npy
│       ├── poster_embeddings_clip.npy / poster_ids_clip.npy
│       └── poster_checkpoint_clip.npz
├── outputs/                    # saved plots
│   ├── poster_pca_scree.png
│   ├── shap_bar.png / shap_beeswarm.png
│   ├── ablation_r2_bar.png
│   ├── baseline_vs_improved.png
│   ├── optuna_history.png
│   ├── error_distribution.png
│   ├── temporal_split.png
│   └── eda_*.png
├── model_comparison_full.png   # full model comparison chart
├── residual_diversity.png      # residual correlation scatter plot
└── catboost_info/              # catboost training artifacts
```

---

## Data Sources

| Source | File | Contents |
|---|---|---|
| TMDB (Kaggle) | `TMDB_movie_dataset_v11.csv` | Budget, revenue, genres, overview, poster URL, language, production country — ~1.4M entries |
| IMDb non-commercial | `title.crew.tsv.csv` | Director and writer IDs per title |
| IMDb non-commercial | `title.principals.tsv.csv` | Top-billed cast per title |
| IMDb non-commercial | `name.basics.tsv.csv` | Resolves person IDs to names |

TMDB provides financial and textual data; IMDb provides structured, reliable talent attribution. Neither source alone is sufficient — TMDB's talent fields are inconsistent text strings, while IMDb has clean person IDs but no financial data.

---

## Pipeline Walkthrough

The notebook is divided into 33 numbered sections. Below is a section-by-section summary grouped by phase.

### Phase 1 — Data (Sections 2–6)

| Section | What it does |
|---|---|
| 2 | **Data acquisition** — loads TMDB CSV and three IMDb TSV files |
| 3 | **Pre-cleaning EDA** — inspects missingness, duplicates, financial outliers, runtime/language/country distributions before any filtering |
| 4 | **Data merging** — left-joins TMDB to IMDb on `imdb_id → tconst`; resolves `nconst` person IDs to names for directors and top cast |
| 5 | **Data cleaning** — filters to released non-adult films with valid budget/revenue; removes placeholder zeros; enforces structural completeness (no missing title/date/genre); defines regression and classification targets |
| 6 | **Post-cleaning EDA** — validates distributions on the cleaned modeling frame |

### Phase 2 — Split & Leakage (Sections 7–8)

| Section | What it does |
|---|---|
| 7 | **Temporal train/val/test split** — sorts by `release_date` and cuts chronologically. This simulates real deployment: the model is trained on historical films and predicts future ones. Random splitting would allow training on 2019 data to predict 2018 films, which is unrealistic. |
| 8 | **Data leakage analysis** — identifies post-release signals (`vote_average`, `vote_count`, `popularity`) that inflate R² artificially. A demonstration trains two Ridge models (with/without these columns) to quantify the inflation. These columns are excluded from all real features. |

### Phase 3 — Feature Engineering (Sections 9–10)

| Modality | Features | Dimensionality |
|---|---|---|
| Structured metadata | `log_budget`, `runtime`, `release_month`, `release_year`, `is_english` | ~5 |
| Genre indicators | One-hot encoded genres (vocabulary fixed on train set) | ~20 |
| Talent features | Expanding median of director's and lead cast's prior films' log-revenue (ordered by release date; median used over mean for robustness to breakout hits) | 2 |
| Synopsis embeddings | `all-MiniLM-L6-v2` sentence transformer; cached to disk | 384-dim |
| Poster embeddings | ResNet18 (classification head removed) applied to downloaded TMDB poster images; cached to disk. Replaced by CLIP (`openai/clip-vit-base-patch32`) in the improvements phase — see below. | 512-dim |

Section 10 **fuses all modalities** by horizontal concatenation into one design matrix (early fusion), allowing any downstream model to learn cross-modal interactions.

### Phase 4 — Baseline & Evaluation (Sections 11–17)

| Section | What it does |
|---|---|
| 11 | **Baseline model** — Ridge regression on structured metadata only; establishes the performance floor |
| 12 | **Evaluation metrics** — defines helpers used throughout: RMSE, MAE, R² for regression; precision, recall, F1, AUC for profitability classification |
| 13 | **Model selection + LightGBM** — compares Ridge vs LightGBM on the full feature matrix; LightGBM uses early stopping (patience = 50 rounds) on the val set |
| 14 | **Ablation study** — adds modalities one at a time (structured → +genre → +talent → +synopsis → +poster) using a fixed Ridge model so only the features vary |
| 15 | **Interpretability with SHAP** — `TreeExplainer` on the LightGBM model; produces bar and beeswarm plots of feature importance |
| 16 | **Error analysis** — identifies films with the largest prediction errors to understand systematic failure modes |
| 17 | **Final evaluation on test set** — test set used exactly once; all model selection and hyperparameter decisions were made on the val set only |

### Phase 5 — Improvements (Sections 18–22)

Four targeted improvements over the baseline LightGBM:

1. **Inflation-adjusted budget** — applies approximate US CPI indices (anchored to 2020 = 100) so the model reasons about real spending power rather than nominal dollars. This addresses the val/test RMSE gap caused by distribution shift over time.
2. **CLIP poster embeddings + PCA** — replaces ResNet18 with `openai/clip-vit-base-patch32`. ResNet18 was pretrained on ImageNet object categories (cats, cars, furniture) and does not capture cinematic signals; CLIP was trained on image–text pairs and learns visual features that align with language and genre concepts. The resulting 512-dim CLIP vectors are then reduced to 50 components via PCA (~80% variance retained) to remove noise and prevent the poster dimensions from drowning out more informative features.
3. **Upgraded synopsis embeddings** — replaces `all-MiniLM-L6-v2` (384-dim, speed-optimised) with `all-mpnet-base-v2` (768-dim, higher accuracy on semantic benchmarks). Cached separately.
4. **Optuna hyperparameter tuning** — Bayesian optimisation (TPE sampler) over 50 trials, each training LightGBM with early stopping on the val set; objective is to minimise val RMSE.
5. **Dedicated profitability classifier** — a separate LightGBM classifier trained directly on the binary `profitable` label, instead of thresholding predicted log-revenue.

Sections 20–22 evaluate the improved model on the test set and produce a side-by-side comparison table with delta and % change columns.

### Phase 6 — Ensemble (Sections 23–33)

**Motivation (Section 23):** The LightGBM v2 model reaches test R² = 0.662 / RMSE = 1.749. A single model learns one set of decision boundaries and cannot hedge its own blind spots. Combining models whose errors are not perfectly correlated can push past this ceiling.

| Section | What it does |
|---|---|
| 24 | Adds XGBoost and CatBoost to the environment; creates a combined train+val pool for OOF stacking |
| 25 | **XGBoost** — level-wise tree growth; second-order gradients; Optuna-tuned (50 trials). Val RMSE ≈ 1.582, nearly tied with LightGBM v2. |
| 26 | **CatBoost** — ordered boosting (eliminates within-round target leakage); symmetric trees; Optuna-tuned |
| 27 | **Residual diversity analysis** — computes pairwise residual correlations between the three models. Low correlation confirms the models make different errors, justifying ensembling. |
| 28 | **Stacking ensemble** — 5-fold out-of-fold (OOF) meta-learner. Base models generate OOF predictions on train+val; a Ridge meta-learner is trained on these. OOF is used instead of direct val-set stacking to prevent the meta-learner from overfitting to a single held-out fold. |
| 29 | **Weighted average ensemble** — `scipy.optimize.minimize` finds the convex combination of the three models' predictions that minimises val RMSE; simpler and interpretable alternative to stacking. |
| 30 | **Final ensemble test evaluation** — both ensemble strategies evaluated on the test set (used once) |
| 31 | **Full model comparison** — table and visualisation covering every model from Ridge baseline through all three ensembles |
| 32 | **Poster embedding ablation** — ablates poster features specifically on LightGBM to answer whether ResNet18 embeddings help or add noise |
| 33 | **Final pitfalls checklist** — automated assertions verify data integrity guarantees across all models |

---

## Models Summary

| Model | Type | Key characteristic |
|---|---|---|
| Ridge | Linear | Baseline; structured features only |
| LightGBM | Gradient boosting | Leaf-wise growth; early stopping |
| LightGBM v2 | Gradient boosting | + CPI budget, PCA poster, mpnet synopsis, Optuna tuning |
| LightGBM classifier | Gradient boosting | Dedicated binary profitability classifier |
| XGBoost | Gradient boosting | Level-wise growth; second-order gradients |
| CatBoost | Gradient boosting | Ordered boosting; symmetric trees |
| Stacking ensemble | Meta-learner | 5-fold OOF Ridge over the three boosters |
| Weighted ensemble | Convex combination | Scipy-optimised weights over the three boosters |

---

## Setup & Running

```bash
# create a virtual environment and install dependencies
pip install -r requirements.txt

# open the notebook
jupyter notebook notebook.ipynb
```

> **Note:** All pre-computed embedding caches (`.npy` files) are committed to the repository under `data/processed/`. This includes both synopsis embeddings (MiniLM and MPNet) and poster embeddings (CLIP). No re-computation is needed — the notebook loads them from disk automatically.

---

## Key Results

| Model | Val RMSE | Test RMSE | Test R² |
|---|---|---|---|
| Ridge (baseline) | — | — | — |
| LightGBM (full features) | ~1.60 | ~1.78 | ~0.64 |
| LightGBM v2 (improved) | ~1.58 | 1.749 | 0.662 |
| XGBoost | ~1.582 | — | — |
| Ensemble (best) | — | see Section 30 | see Section 30 |

All metrics are on `log1p(revenue)` — lower RMSE and higher R² are better. The test set is evaluated exactly once per model to prevent information leakage.
