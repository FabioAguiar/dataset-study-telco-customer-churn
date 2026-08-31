# Telco Customer Churn — Dataset Study

End-to-end, reproducible study of the **Telco Customer Churn** dataset. The project covers source acquisition, structural validation, exploratory analysis, deterministic preparation, model-family comparison, hyperparameter search, final holdout evaluation, model bundling, and a controlled educational inference demonstration.

The analysis is predictive rather than causal. It measures whether observed customer-account characteristics contain signal for distinguishing `Churn = Yes` from `Churn = No`; it does not establish that changing any individual feature would change future churn behavior.

## At a glance

| Item | Result |
|---|---:|
| Observation unit | Customer account |
| Source rows | 7,043 |
| Source columns | 21 |
| Model features | 19 |
| Identifier excluded from modeling | `customerID` |
| Target | `Churn` |
| Positive class | `Yes` |
| Positive-class prevalence | 26.54% |
| Problem type | Binary classification |
| Primary model-selection metric | Average Precision |
| Selected model | HistGradientBoostingClassifier |
| Validation Average Precision | 0.6708 |
| Final-test Average Precision | 0.6413 |
| Final-test ROC-AUC | 0.8402 |
| Frozen educational threshold | 0.2578 |
| Operational prediction available | No |

## Study objective

The study asks whether customer profile, service, contract, billing, payment, tenure, and charge information can predict the probability of `Churn = Yes` under a controlled random-snapshot evaluation protocol.

The workflow is designed to keep analytical decisions visible while moving reusable validation, preparation, model-selection, finalization, and inference logic into tested Python modules.

The study aims to:

- understand the dataset, target, feature roles, and quality constraints;
- preserve immutable raw data and deterministic preparation rules;
- prevent identifier and target leakage;
- isolate train, validation, and test partitions;
- compare multiple model families under one feature and evaluation contract;
- expose the actual hyperparameter search policy used for every candidate family;
- select the final candidate without consulting the sealed test partition;
- evaluate the selected model exactly once on final test;
- serialize preprocessing and estimator logic as one fitted pipeline;
- validate artifact integrity and runtime compatibility before deserialization;
- demonstrate inference without claiming operational readiness.

## Dataset and source

The study uses the Kaggle dataset [`blastchar/telco-customer-churn`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Each row represents one customer account.

| Role | Columns |
|---|---|
| Identifier | `customerID` |
| Target | `Churn` |
| Numerical features | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Categorical features | 16 customer, service, contract, billing, and payment fields |
| Positive class | `Yes` |
| Negative class | `No` |

The model excludes `customerID` and never receives `Churn` as an input feature.

Download the source through the project acquisition utility:

```bash
python -m scripts.download_data kaggle \
  blastchar/telco-customer-churn \
  --destination data/raw/telco-customer-churn
```

Raw and generated datasets are intentionally excluded from version control.

## Data quality and preparation

The source contains 7,043 unique customer accounts and no duplicated customer identifiers.

The principal source-quality issue is limited to `TotalCharges`:

- 11 raw values are blank;
- every blank occurs where `tenure == 0`;
- those values are deterministically materialized as `0.0`;
- no row is removed;
- no mean, median, mode, or learned imputation rule is introduced for this correction.

The prepared snapshot is split reproducibly using stratification and random seed `42`:

| Partition | Rows | Purpose |
|---|---:|---|
| Train | 4,930 | Cross-validation and hyperparameter search |
| Validation | 1,056 | Candidate comparison and educational threshold selection |
| Test | 1,057 | One-time final evaluation only |
| Final fit | 5,986 | Train plus validation after selection is frozen |

The test partition remains sealed during feature decisions, hyperparameter search, model selection, and threshold selection.

Categorical variables are encoded inside the fitted pipeline with `OneHotEncoder(handle_unknown="ignore")`. Numerical scaling is applied only to Logistic Regression during candidate search; the selected HistGradientBoosting pipeline uses numerical passthrough.

## Exploratory evidence

Exploratory results describe **associations in this dataset snapshot**. They must not be interpreted as causal effects.

### Target distribution

The target contains 5,174 `No` observations and 1,869 `Yes` observations.

| Churn class | Observations | Share |
|---|---:|---:|
| `No` | 5,174 | 73.46% |
| `Yes` | 1,869 | 26.54% |

The positive class is sufficiently less frequent than the negative class that accuracy alone would provide an incomplete model-selection signal. The study therefore prioritizes Average Precision and records additional probability and threshold-dependent metrics.

![Distribution of the Churn target classes](docs/images/churn_target_class_distribution.png)

### Contract term is strongly associated with observed churn

Observed churn rates differ substantially across contract categories:

| Contract | Observed churn rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.27% |
| Two year | 2.83% |

![Observed churn rate by contract category](docs/images/contract_churn_rate_by_category.png)

Month-to-month accounts show much higher observed churn than accounts on longer contracts. This is predictive evidence, not proof that changing contract type would independently prevent churn.

### Churn is concentrated in shorter observed relationships

Across tenure quantiles, observed churn decreases from approximately 58.4% in the earliest-tenure group to approximately 3.5% in the latest.

![Observed churn rate by tenure quantile](docs/images/tenure_churn_rate_by_quantile.png)

Tenure contains substantial predictive signal, but it is also mechanically related to how long an account has already remained active. The relationship is therefore not a causal retention effect.

### Service, support, and billing variables add additional signal

The strongest categorical associations with churn include contract type, online security, technical support, internet service, and payment method.

![Ranking of categorical feature associations with churn](docs/images/feature_to_target_categorical_association_ranking.png)

These rankings describe association strength in the observed sample. They do not identify intervention effects.

## Evaluation protocol

Model selection is performed only with train and validation data.

| Component | Contract |
|---|---|
| Evaluation mode | Stratified random snapshot |
| Primary selection metric | Average Precision |
| Cross-validation | 5-fold `StratifiedKFold` |
| CV shuffle | `True` |
| CV random seed | `42` |
| Search refit metric | Average Precision |
| Dummy eligibility margin | Candidate AP must exceed Dummy AP by more than `0.01` |
| Practical-tie tolerance | Validation AP difference `<= 0.01` with overlapping approximate CV intervals |
| Threshold-selection partition | Validation |
| Final-test use before selection | Prohibited |

Average Precision is primary because the positive class is the minority class. ROC-AUC, precision, recall, F1, F2, balanced accuracy, Brier Score, and Log Loss are retained as complementary evidence.

## Model selection

### Candidate comparison

The Dummy prior classifier establishes a non-eligible baseline. Four model families are eligible for selection.

The table below reports the best cross-validation result produced by each candidate search and the corresponding one-time validation evaluation at threshold `0.50`.

| Model | Search | CV AP mean ± std | Validation AP | Validation ROC-AUC | Validation Brier ↓ | Selection status |
|---|---|---:|---:|---:|---:|---|
| HistGradientBoostingClassifier | RandomizedSearchCV | **0.6728 ± 0.0161** | **0.6708** | **0.8477** | **0.1332** | **Selected** |
| Logistic Regression | GridSearchCV | 0.6591 ± 0.0129 | 0.6688 | 0.8470 | 0.1339 | Practical-tie finalist |
| Random Forest | RandomizedSearchCV | 0.6659 ± 0.0123 | 0.6679 | 0.8475 | 0.1593 | Eligible candidate |
| Decision Tree | GridSearchCV | 0.6194 ± 0.0234 | 0.6134 | 0.8161 | 0.1462 | Eligible candidate |
| Dummy prior | No search | — | 0.2652 | 0.5000 | 0.1948 | Baseline only |

HistGradientBoosting and Logistic Regression satisfy the study's practical-tie condition on validation Average Precision. The predefined first decisive tie-break is lower validation Brier Score, which favors HistGradientBoosting (`0.133203` versus `0.133898`).

### Candidate search configuration and hyperparameters

The search policy is frozen before validation evaluation. The values below are the **actual hyperparameter spaces evaluated by Notebook 03**, not generic examples.

| Model | Search policy | Evaluated configurations | Fixed estimator settings | Hyperparameters explored |
|---|---|---:|---|---|
| Logistic Regression | GridSearchCV | 24 | `solver=liblinear`; `max_iter=2000`; `random_state=42`; numerical `StandardScaler` | `C={0.001,0.01,0.1,1,10,100}`; `penalty={l1,l2}`; `class_weight={None,balanced}` |
| Decision Tree | GridSearchCV | 48 | `random_state=42` | `criterion={gini,entropy}`; `max_depth={3,5,8,None}`; `min_samples_leaf={1,10,30}`; `class_weight={None,balanced}` |
| Random Forest | RandomizedSearchCV | 40 | `random_state=42`; estimator `n_jobs=1`; search `n_jobs=4` | `n_estimators={300,500,800}`; `max_depth={None,8,12,20}`; `min_samples_split={2,10,20}`; `min_samples_leaf={1,2,5,10}`; `max_features={sqrt,0.5,None}`; `class_weight={None,balanced,balanced_subsample}` |
| HistGradientBoosting | RandomizedSearchCV | 40 | `random_state=42`; search `n_jobs=4` | `learning_rate={0.03,0.05,0.1,0.2}`; `max_iter={100,200,400}`; `max_leaf_nodes={7,15,31,63}`; `max_depth={None,3,5,8}`; `min_samples_leaf={10,20,40}`; `l2_regularization={0,0.01,0.1,1,10}` |
| Dummy prior | No search | 1 | `strategy=prior` | None |

The generated model-selection artifacts preserve the detailed search outcomes for runtime auditability. The downstream handoff freezes only the configuration selected for finalization.

### Selected configuration

The winning HistGradientBoosting configuration is:

| Hyperparameter | Selected value |
|---|---:|
| `learning_rate` | 0.03 |
| `max_iter` | 200 |
| `max_depth` | 3 |
| `max_leaf_nodes` | 7 |
| `min_samples_leaf` | 40 |
| `l2_regularization` | 1.0 |
| `random_state` | 42 |

The final serialized object is a complete scikit-learn `Pipeline` containing:

- a `ColumnTransformer`;
- numerical passthrough;
- a fitted `OneHotEncoder(handle_unknown="ignore")` for categorical features;
- the fitted `HistGradientBoostingClassifier`.

No external preprocessing step is required during inference.

## Final holdout evaluation

After model selection and educational-threshold selection are frozen, the selected pipeline is fitted once on train plus validation data and evaluated exactly once on the sealed test partition.

| Metric | Validation | Final test | Test − validation |
|---|---:|---:|---:|
| Average Precision | 0.6708 | **0.6413** | -0.0295 |
| ROC-AUC | 0.8477 | **0.8402** | -0.0076 |
| Brier Score ↓ | 0.1332 | **0.1394** | +0.0062 |
| Log Loss ↓ | 0.4135 | **0.4207** | +0.0072 |

The holdout retains useful ranking performance, with a moderate reduction in Average Precision and no evidence of a performance collapse within the same random-snapshot contract.

These results do **not** establish temporal generalization, prospective performance, calibration adequacy for production, or intervention effectiveness.

## Threshold and decision-policy diagnostics

Threshold selection is performed only on validation data. The frozen educational rule maximizes precision subject to validation recall of at least `0.80`.

The selected educational threshold is:

```text
0.2577809673219062
```

It is an educational decision rule, not a business-optimal operating point.

| Final-test result | Threshold 0.50 | Educational threshold 0.2578 |
|---|---:|---:|
| Precision | 62.50% | 51.25% |
| Recall | 49.82% | 80.43% |
| F1 | 55.45% | 62.60% |
| F2 | 51.93% | 72.20% |
| Balanced accuracy | 69.50% | 76.36% |
| True positives | 140 | 226 |
| False negatives | 141 | 55 |
| False positives | 84 | 215 |
| Predicted positives | 224 | 441 |

Relative to threshold `0.50`, the lower educational threshold identifies 86 additional positive test cases while producing 131 additional false positives. An operational threshold would require customer value, intervention cost, campaign capacity, and asymmetric error-cost information that this dataset does not provide.

## Inference contract and demonstration

Notebook 05 demonstrates trusted local inference using synthetic inputs created in memory.

The inference flow validates:

1. final-model handoff integrity;
2. inference-bundle integrity;
3. educational readiness and non-operational flags;
4. artifact path safety;
5. model file existence and SHA-256;
6. handoff, manifest, and bundle alignment;
7. runtime compatibility before deserialization;
8. explicit `trusted_source=True`;
9. fitted pipeline structure and state;
10. input schema, missing-value policy, unknown categories, and output semantics.

The demonstration supports a mapping, pandas Series, or one/multiple-row pandas DataFrame and returns positive-class probabilities plus the educational threshold classification.

It does not:

- use train, validation, or test observations as inference examples;
- call `fit` or `fit_transform`;
- persist customer inputs, probabilities, or predictions;
- expose an operational API;
- establish production validity.

Every demonstration result preserves:

```text
operational_prediction_available = false
```

## Workflow and notebooks

```text
Raw dataset
    ↓
01 — Understanding and exploratory analysis
    ↓
02 — Deterministic preparation and partitioning
    ↓
03 — Candidate search, model comparison, and threshold analysis
    ↓
04 — Frozen final fit, one-time test evaluation, and model bundle
    ↓
05 — Trusted educational inference demonstration
```

| Notebook | Responsibility |
|---|---|
| [`01_data_understanding_and_exploration.ipynb`](notebooks/01_data_understanding_and_exploration.ipynb) | Dataset context, quality validation, EDA, leakage review, and preparation decisions |
| [`02_data_preparation.ipynb`](notebooks/02_data_preparation.ipynb) | Deterministic correction, feature contract, stratified partitioning, and preparation handoff |
| [`03_model_selection_and_evaluation.ipynb`](notebooks/03_model_selection_and_evaluation.ipynb) | Dummy baseline, candidate searches, CV evidence, validation comparison, selection, and educational threshold |
| [`04_final_model_and_bundle.ipynb`](notebooks/04_final_model_and_bundle.ipynb) | Frozen final fit, one-time final-test evaluation, serialization, manifests, and inference bundle |
| [`05_inference_demo.ipynb`](notebooks/05_inference_demo.ipynb) | Runtime gate, trusted loading, input normalization, and educational inference examples |

## Reproducibility

### Environment setup

Create or activate a Python 3.10+ environment and install the project from the repository root:

```bash
python -m pip install -e ".[notebook,test]"
```

Optional Jupyter kernel registration:

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

### Recorded serialized-model runtime

The final inference bundle records the runtime used to create and validate the model artifact:

| Component | Recorded version |
|---|---:|
| Python | 3.13.13 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| joblib | 1.5.3 |

The educational loader checks runtime compatibility before deserializing the joblib artifact.

### Reproducing the notebooks

Run the notebooks in numerical order from a fresh kernel. Each stage validates the persisted handoff produced by the previous stage.

```bash
for notebook in \
  notebooks/01_data_understanding_and_exploration.ipynb \
  notebooks/02_data_preparation.ipynb \
  notebooks/03_model_selection_and_evaluation.ipynb \
  notebooks/04_final_model_and_bundle.ipynb \
  notebooks/05_inference_demo.ipynb
do
  python -m jupyter nbconvert \
    --to notebook \
    --execute "$notebook" \
    --ExecutePreprocessor.timeout=-1 \
    --inplace
done
```

Before Notebook 05, the process runtime must satisfy the compatibility contract stored in the inference bundle.

### Tests

Run the reusable test suite with:

```bash
PYTHONPATH=. python -m pytest
```

Run the inference tests separately with:

```bash
PYTHONPATH=. python -m pytest tests/test_smoke_predict.py
```

Compile-check the inference module with:

```bash
python -m py_compile scripts/smoke_predict.py
```

## Repository structure

```text
.
├── api/                  Reserved future runtime/API scaffold
├── artifacts/            Runtime-generated manifests and model artifacts
├── data/                 Raw, interim, processed, and external data areas
├── docs/images/          Exported exploratory evidence
├── notebooks/            Dataset-specific analytical narrative and decisions
├── scripts/              Reusable validation, analysis, preparation, and inference logic
├── tests/                Unit tests for reusable modules and contracts
├── pyproject.toml         Package metadata and dependency groups
└── README.md              Scientific project overview and selected evidence
```

Generated data, runtime JSON/CSV evidence, serialized models, caches, environments, and credentials are excluded from version control.

## Reproducibility and integrity controls

The workflow records and validates:

- feature and target contracts;
- partition paths, row counts, class counts, and SHA-256 hashes;
- artifact byte hashes and semantic fingerprints;
- model-search strategies and parameter spaces;
- selected model and frozen hyperparameters;
- educational-threshold origin;
- final-test access count;
- runtime versions;
- model-state fingerprint;
- trusted-source confirmation before model deserialization.

These controls make the study auditable without treating generated runtime artifacts as source code.

## Limitations and readiness

| Capability | Status |
|---|---|
| Dataset understanding and EDA | Completed |
| Deterministic preparation | Completed |
| Multi-family model selection | Completed |
| Final model training | Completed |
| One-time final-test evaluation | Completed |
| Model artifact and inference bundle | Materialized at runtime |
| Educational inference demonstration | Completed in the recorded compatible runtime |
| Operational modeling validity | Unconfirmed |
| Operational threshold | Unresolved |
| Temporal validity | Unresolved |
| Production feature availability | Unconfirmed |
| API implementation | Not implemented |
| Operational prediction | Unavailable |

Additional limitations:

- the evaluation uses a stratified random snapshot rather than a temporal or prospective holdout;
- observed associations and model importance are not causal effects;
- future distribution stability has not been evaluated;
- the dataset's external representativeness is not established;
- business costs for false positives and false negatives are unavailable;
- no intervention-uplift or retention-effectiveness analysis was performed;
- no subgroup fairness assessment is established;
- no production drift-monitoring or retraining policy is evaluated;
- the educational threshold is not a validated retention policy;
- the study does not establish the safety or effectiveness of automated customer decisions.

## Responsible interpretation

The study demonstrates useful predictive structure for distinguishing customer accounts associated with `Churn = Yes` in this dataset snapshot. Contract term, tenure, charges, internet service, and related service variables contain meaningful predictive information, while HistGradientBoosting and Logistic Regression perform similarly under the defined validation protocol.

HistGradientBoosting is selected through the predefined Brier-score tie-break and achieves final-test Average Precision `0.6413` and ROC-AUC `0.8402` in the study's sealed random holdout.

These results support an educational binary-classification benchmark. They do not establish that changing any individual feature would reduce churn, that the educational threshold is economically optimal, or that the model is ready for autonomous retention decisions.

For exhaustive analysis, inspect the notebooks and the complete evidence set under [`docs/images/`](docs/images/).
