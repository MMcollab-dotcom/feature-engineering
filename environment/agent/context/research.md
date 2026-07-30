# Public research guide

The objective is stable out-of-sample prediction of `target_horizon_1`, followed
by credible after-cost portfolio performance. Treat the target as an opaque
supervised response and every supplied market column as an unvalidated input.
Public datetimes are anonymized synthetic UTC coordinates: ordering and elapsed
durations are real, but calendar labels do not identify historical events.

Form testable hypotheses for causal features, then run the informative
experiments that can support or reject them.

Begin with chronological and per-symbol data checks. Keep every symbol at a
given datetime in the same fold: fit on earlier datetimes and compare candidates
on later chronological blocks. Fit every learned preprocessing step—including
imputation, scaling, clipping or winsorization limits, thresholds, and
selectors—on training rows only. Evaluate Pearson IC, Spearman IC, weighted
error, sign stability, prediction dispersion, and per-symbol behavior.

Causal transformations may use the current and earlier rows within each symbol,
including lags, rolling windows, and expanding histories. A prediction at
datetime `t` must never depend on a later row. In particular, do not use centered
windows, negative shifts, backward filling, or statistics learned from the full
prediction batch. Prediction is audited by perturbing future suffixes: changing
later input rows must leave every earlier prediction unchanged.

Build an informative feature panel. Reject weak, unstable, redundant, or
outlier-driven candidates. Use `weight_std_dollar_vol` as `sample_weight` for
fitting and weighted losses, but exclude it from predictors.

Standardize model inputs inside the fitted pipeline. Compare a small
neighborhood of viable positive `alpha` and `l1_ratio` values rather than relying
on one arbitrary setting or a broad grid. Reject settings that produce all-zero
coefficients, constant predictions, or effectively unregularized unstable fits.

Training and backtesting share 100 research attempts. Malformed requests and
other protocol errors share a separate 10-error allowance. Accepted training or
backtest work consumes one research attempt; tool results report the remaining
counts.

The public columns are:

- identity: `datetime`, `symbol`
- predictors to evaluate: `open`, `high`, `low`, `close`, `volume`,
  `quote_asset_volume`, `number_of_trades`,
  `taker_buy_base_asset_volume`, `taker_buy_quote_asset_volume`
- sample weight only: `weight_std_dollar_vol`
- target: `target_horizon_1`

Use scratch scripts or notes as needed, but the submitted estimator must be
self-contained and depend only on the `X` and `y` passed to `train_model`.
