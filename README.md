# Adult Census Income – ML pipeline

Predicts whether a person earns over $50K/year from 1994 US census attributes.

## Layout
```
src/
  config.py    paths, column names, split sizes
  data.py      load (OpenML or local UCI file), clean, stratified split
  features.py  hand-written Preprocessor: impute, scale, group rare categories, one-hot
  models.py    candidate models (receive preprocessed features)
  evaluate.py  accuracy, F1, ROC AUC, PR AUC
  train.py     train → select on validation → test once → save
  predict.py   load saved model and score new rows
```

## Run
Before you run this, if you are on MacOS, run this command to ensure you don't run into SSL certificate errors: 

```bash
open "/Applications/Python 3.14/Install Certificates.command"
```
```bash
pip install -r requirements.txt
python -m src.train      # downloads data on first run (~5 MB), trains in seconds
python -m src.predict    # scores a built-in example
```

## Design notes
- Preprocessing is implemented by hand. All statistics
  are fit on the training split only and reused for val/test/inference; the fitted preprocessor
  is saved alongside the model in `artifacts/model.joblib`.
- Train / val / test split (64/16/20). Model selection uses validation only; test is touched once.
- `fnlwgt` (census sampling weight) and `education` (duplicate of `education_num`) are dropped.
- No hyperparameter tuning: logistic regression baseline vs. histogram gradient boosting with defaults.
- Classes are imbalanced (~24% positive), so F1 and PR AUC are reported alongside accuracy.

## Results

Dataset after cleaning and de-duplication: **42,468** rows (train **27,179** / val **6,795** / test **8,494**),
**24.6%** positive (>50K). Preprocessing produces **50** features.

### Validation (model selection)
| Model | Accuracy | F1 | ROC AUC | PR AUC | Train time |
|---|---|---|---|---|---|
| Logistic regression | 0.849 | 0.664 | 0.902 | 0.773 | 0.1s |
| **Hist gradient boosting** | **0.872** | **0.722** | **0.925** | **0.837** | 1.3s |

### Test (selected model: hist gradient boosting, evaluated once)
| Accuracy | F1 | ROC AUC | PR AUC |
|---|---|---|---|
| 0.869 | 0.714 | 0.924 | 0.829 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| <=50K | 0.895 | 0.936 | 0.915 | 6,404 |
| >50K | 0.771 | 0.665 | 0.714 | 2,090 |

### Takeaways
- Gradient boosting beat the logistic regression baseline on every validation metric
  (ROC AUC +0.024, PR AUC +0.063, F1 +0.058) with no tuning, at ~1 second of training.
- Test scores are within ~0.01 of validation scores, so selecting on the validation set
  didn't produce an optimistic estimate.
- Accuracy is the least informative metric here: always predicting <=50K scores 75.4% on the
  test set. The model's 86.9% is a real but smaller gain than the raw number suggests.
- The model is conservative on the minority class: when it predicts >50K it is right 77% of
  the time, but it catches only 66.5% of actual >50K earners (~700 missed). Lowering the 0.5
  decision threshold (picked on validation) would trade precision for recall.
