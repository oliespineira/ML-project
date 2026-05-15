# Film revenue prediction

By:

Primary artifact: [`notebook.ipynb`](notebook.ipynb) — film revenue prediction (regression on `log1p(revenue)` and related classification targets).

## Contents

Numbered `##` headings in the notebook follow this order. Bullets are `###` subsections inside the same topic. **4.1–4.3** are milestones inside one code cell (comment headers), not separate markdown cells.

### 1) project overview

- 1.1 central research question
- 1.2 prediction target
- 1.3 core thesis

### 2) data acquisition

- 2.1 tmdb (kaggle)
- 2.2 imdb non-commercial files
- 2.3 poster images

### 3) pre-cleaning eda

- pre-cleaning frame
- missing values
- duplicates
- financial outliers
- runtime distribution
- language distribution
- top production countries
- pre-cleaning eda conclusions

### 4) data merging

- **4.1** merge tmdb + crew (directors & writers)
- **4.2** merge top-billed cast from principals
- **4.3** resolve nconst → name for director and top cast

### other aspects to take into account

### 5) data cleaning

- diagnose missing financial data
- scope the modeling population
- keep valid financial signals
- enforce structural completeness
- define supervised targets

### 6) post-cleaning eda

### 7) train / val / test split

### 8) data leakage analysis

### 9) feature engineering

- 9.1 structured metadata
- 9.2 genre indicators
- 9.3 synopsis embeddings
- 9.4 poster embeddings
- 9.5 talent features (incl. `##` block: director & lead-cast historical revenue)

### 10) multimodal fusion

### 11) baseline model

### 12) evaluation metrics

### 13) model selection + lightgbm

### 14) ablation study

### 15) interpretability with shap

### 16) error analysis

### 17) final evaluation on test set

### 18) pitfalls checklist

### 19) improvements

- 18.1 inflation-adjusted budget
- 19.2 poster pca dimensionality reduction
- 19.3 upgraded synopsis embeddings
- 19.4 fuse improved feature matrices
- 19.5 optuna hyperparameter tuning
- 18.6 dedicated profitability classifier

### 20) improved model — final evaluation on test set

### 21) baseline vs improved — full comparison

### 22) pitfalls checklist

### 23) motivation: why ensembles work

- 23.2 The bias-variance decomposition
- 23.3 Why XGBoost and CatBoost are genuinely different from LightGBM
- 23.4 Two ensemble strategies

### 24) install & import new libraries

- 24.1 — Analysis: Install step
- 24 — Analysis: Train+val pool

### 25) XGBoost Regressor

- 25.1 What XGBoost does differently
- 25.2 Hyperparameter search design
- 25.2 — Analysis: Optuna search results
- 25.3 — Analysis: XGBoost val performance

### 26) CatBoost Regressor

- 26.1 What CatBoost does differently
- 26.2 Hyperparameter design
- 26.2 — Analysis: CatBoost Optuna search results
- 26.3 — Analysis: Three-model individual comparison

### 27) Model Diversity Analysis — Correlation of Residuals

- 27.1 Why this check must come before ensembling
- 27.2 — Analysis: Residual correlations
- 27.3 — Analysis: Scatter plot interpretation

### 28) Stacking Ensemble — Out-of-Fold Meta-Learner

- 28.1 Why OOF stacking, not direct val-set stacking
- 28.2 — Analysis: OOF stability and expected performance
- 28.3 — Analysis: Meta-learner weights and regularisation
- 28.4 — Analysis: Full retrain design choices

### 29) Weighted Average Ensemble

- 29.1 Why a simpler alternative
- 29.2 — Analysis: Optimal weights

### 30) Ensemble — Final Test Evaluation

- 30.1 Protocol
- 30.2a — Analysis: Individual model test performance
- 30.2 — Analysis: Test set results

### 31) Full Model Comparison Table & Visualisation

- 31.1 The complete learning arc
- 31.1a — Analysis: Reading the comparison table
- 31.2 — Analysis: The full learning arc

### 32) Poster Embedding Ablation — Drop vs Keep

- 32.1 The open question from notebook2
- 32.2 Why ResNet18 poster features may not help
- 32.3 Why we test on LightGBM specifically
- 32.4 — Analysis: What the result tells us

### 33) Final Pitfalls Checklist — Notebook 3

- 33.2 — Project Summary & Conclusions
