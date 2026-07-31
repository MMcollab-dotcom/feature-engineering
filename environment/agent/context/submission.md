# Submitted model contract

Submit Python source defining exactly one entry point:

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline

def train_model(X, y):
    features = ["close", "volume"]  # replace with evidence-backed inputs
    weight = X["weight_std_dollar_vol"].astype("float64")
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        ElasticNet(alpha=1e-5, l1_ratio=0.5, max_iter=10000),
    )
    model.fit(X.loc[:, features], y.to_numpy().ravel(),
              elasticnet__sample_weight=weight)
    return model
```

The example shows the interface, not recommended features or hyperparameters.
Every submitted candidate must return either an exact
`sklearn.linear_model.ElasticNet` or an exact `sklearn.pipeline.Pipeline` whose
final step is an exact `ElasticNet`. Subclasses and pipelines ending in another
estimator are rejected. Pass `weight_std_dollar_vol` as fitting sample weight.
The top-level fitted estimator must expose unique ordered string
`feature_names_in_` values.

`X` and `y` share a two-level `datetime`, `symbol` index. Prediction must return
finite numeric values in the same row order. Accepted one-target shapes are
`(n_rows,)` and `(n_rows, 1)`.

Submitted source may import only `math`, `statistics`, `numpy`, `pandas`, and
`sklearn`. It may not read or write files. Maximum source size is 20,000 UTF-8
bytes; every fit and prediction has a 1,800-second deadline. The returned model
must serialize with Joblib, reload in a fresh process, and predict without
calling `train_model` again.

## Official full-public refit

The fitted artifact associated with the submitted public backtest establishes
submission provenance and validates the model contract, but it is not the model
used for the hidden score. During verification, the submitted source is imported
in the isolated model worker and `train_model` is called once with every public
2022–2023 row. The resulting newly fitted estimator is serialized, reloaded,
and used for causal prediction on the hidden 2024 batch before the after-cost
backtest is scored.

The submitted source and all learned preprocessing must therefore support a
single fit over the complete public period. Hidden labels and scoring-only
columns are never attached to the model's prediction input.
