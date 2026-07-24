# Telco Customer Churn Dataset Study

Reproducible dataset study organized around notebooks, reusable Python
utilities, immutable raw data, model artifacts, and a future inference API.

## Project structure

```text
api/          Runtime adapter and inference API
artifacts/    Generated model artifacts
data/         Raw, interim, processed, and external data
notebooks/    Analytical narrative and study-specific decisions
scripts/      Reusable acquisition, context, validation, and workflow logic
tests/        Unit tests for reusable Python utilities
```

The notebooks should keep dataset-specific parameters and analytical results
visible. Reusable operational logic belongs in `scripts/`.

## Environment setup

From the project root, create or activate a Python 3.10+ environment and
install the project in editable mode:

```bash
python -m pip install -e ".[notebook,test]"
```

Editable installation allows notebooks to import `scripts.*` regardless of
whether Jupyter is started from the project root or from `notebooks/`.

Registering a dedicated kernel is optional but recommended:

```bash
python -m ipykernel install \
  --user \
  --name dataset-study-telco \
  --display-name "Python (dataset-study-telco)"
```

Start JupyterLab:

```bash
python -m jupyter lab
```

## Dataset acquisition

The first notebook keeps the Kaggle handle and destination visible while
delegating transport, validation, and file discovery to
`scripts/download_data.py`.

Equivalent command-line acquisition:

```bash
python -m scripts.download_data kaggle \
  blastchar/telco-customer-churn \
  --destination data/raw/telco-customer-churn
```

Raw files are ignored by Git and must not be overwritten by cleaning or
preparation steps.

## Tests

Run the reusable utility tests with:

```bash
python -m pytest
```
