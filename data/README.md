# dataset setup guide

all raw datasets are kept local and are not pushed to github.

## required files

put these files in `data/raw/` with the exact names below:

- `tmdb_movies.csv`
- `title.crew.tsv.gz`
- `title.principals.tsv.gz`
- `name.basics.tsv.gz`

## where to download

- tmdb dataset (kaggle): [tmdb movies dataset 2023](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
- imdb files: [imdb non-commercial datasets](https://datasets.imdbws.com/)

download these imdb files only:

- `title.crew.tsv.gz`
- `title.principals.tsv.gz`
- `name.basics.tsv.gz`

## expected folder structure

```text
data/
  README.md
  raw/
    tmdb_movies.csv
    title.crew.tsv.gz
    title.principals.tsv.gz
    name.basics.tsv.gz
  processed/
```

## quick setup steps

1. create folders:
   - `data/raw`
   - `data/processed`
2. download tmdb csv and move it to `data/raw/tmdb_movies.csv`
3. download the 3 imdb `.tsv.gz` files and move them to `data/raw/`
4. run the notebook loading cell to verify all files are found

## team workflow note

everyone keeps these large files locally. commit code, configs, and documentation only.
