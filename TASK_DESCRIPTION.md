# Causal Feature Engineering for Quantitative Trading

## Scientific Domain

**Domain:** Quantitative Finance  
**Field:** Systematic Trading  
**Subfield:** Machine-Learned Alpha Research

## Scientific Problem

Quantitative researchers rarely succeed by fitting one model and accepting the
largest public-period score. Minute-level market data are noisy, predictive
relationships are weak, regimes change, and apparently strong signals can
disappear after turnover, execution costs, or exposure normalization. The
research problem is to discover causal predictive structure that remains useful
outside the period in which it was designed.

This task compresses that research loop into a fixed, reproducible environment.
The agent is expected to inspect market data, form feature hypotheses, run
controlled chronological experiments, compare predictive and economic
diagnostics, reject unstable candidates, and select one strategy for hidden
future scoring. Public metrics may conflict: a candidate can have attractive
information coefficient but poor after-cost performance, or a high Sharpe on
one interval with weak sign and regime stability.

The final predictive estimator is deliberately fixed to an exact Elastic Net so
that improvements must come primarily from causal feature construction,
preprocessing, selection, regularization, and experimental judgment rather than
from switching to a more powerful model family. Symbol identities, calendar
coordinates, and target semantics are obscured to reduce historical lookup and
force evidence-based research on the supplied panel.

## Workflow Details

The agent phase has two cooperating containers. The `main` container contains
the public Parquet panel, task instructions, and the Python data-science stack.
The `mcp-server` sidecar owns model fitting, artifact storage, public
backtesting, research budgets, and final submission promotion. The agent
interacts with that sidecar through five MCP tools:

- `train_model`
- `get_train_model_result`
- `backtest`
- `get_backtest_result`
- `submit_strategy`

The research workflow has four stages:

1. **Public research.** Inspect the 2022–2023 panel chronologically and by
   symbol. Form hypotheses for past-only transformations such as returns, lags,
   rolling statistics, expanding histories, price-range structure, activity,
   and order-flow imbalance. Keep every symbol at a timestamp in the same
   chronological fold and fit learned preprocessing on training rows only.
2. **Source-bound training.** Send complete Python source defining
   `train_model(X, y)` to the asynchronous training tool. The returned fitted
   object must be an exact `ElasticNet` or an exact sklearn `Pipeline` ending in
   an exact `ElasticNet`. It must use `weight_std_dollar_vol` as fitting weight,
   remain serializable with Joblib, and expose a stable inference schema.
3. **Controlled public backtesting.** Backtest accepted models on public windows
   strictly later than their training windows. Compare candidates on consistent
   periods using information coefficient, error, prediction dispersion,
   turnover, after-cost Sharpe, CAGR, and maximum drawdown. Use later slices to
   challenge apparently strong candidates rather than optimizing one interval.
4. **Candidate selection.** Submit the `strategy_id` from one successful
   backtest after all asynchronous operations are idle. The sidecar promotes one
   immutable bundle containing the selected source, fitted public artifact,
   inference contract, strategy settings, and integrity hashes.

Training and backtesting share a budget of 100 accepted research attempts. One
training operation and one backtest may overlap, but two operations of the same
type may not. This permits deliberate parallel research without allowing
unbounded search.

After the agent exits, Harbor collects the bundle and starts a separate
verifier. The verifier checks bundle integrity, imports the submitted source in
a networkless model worker, and calls `train_model` once on every public
2022–2023 row. It serializes and reloads that newly fitted estimator, predicts
on hidden 2024 feature matrices, audits causal invariance by perturbing future
suffixes, and runs the hidden after-cost backtest. Hidden labels and
scoring-only columns never enter the model worker.

## Dependencies & System Requirements

### Agent-facing model environment

Submitted model source runs on Python 3.12 and may import only `math`,
`statistics`, `numpy`, `pandas`, and `sklearn`. The model must fit and predict
without file I/O, external services, or network access. Source is limited to
20,000 UTF-8 bytes; each fit and prediction operation has a 1,800-second
deadline. The returned estimator must serialize with Joblib and predict after
reload in a fresh process. No GPU is available or required.

The broader agent container includes the pinned Python scientific stack needed
to inspect the Parquet data and communicate with the MCP service. Core versions
are Python 3.12, NumPy 2.5.1, pandas 3.0.5, SciPy 1.18.0, scikit-learn 1.9.0,
PyArrow 25.0.0, Joblib 1.5.3, and FastMCP 3.4.5.

### Task deployment environment

The sample targets Harbor 0.20.0 with Docker Compose and Git LFS. The task
requests four CPU cores, 8 GiB of memory, 32 GiB of storage, and no GPU. The
configured limits are 30 minutes for environment construction, four hours for
the agent, and one hour for verification. The agent environment has public
network access for general research, while submitted source executes in a
networkless, read-only model worker with narrowly controlled imports and file
operations.

## Dataset

The benchmark uses a fixed panel of historical cryptocurrency minute bars for
four anonymized symbols. The canonical source universe is BTCUSDT, DOGEUSDT,
ETHUSDT, and SOLUSDT, but mapped to `symbol_01` through `symbol_04`.
Datetimes are replaced with synthetic UTC coordinates while preserving 
ordering and elapsed durations.

The chronological split is:

- **Public research:** 2022-01-01 through 2023-12-31, containing 4,204,792 panel
  rows.
- **Hidden evaluation:** 2024-01-01 through 2024-12-31, containing 2,108,152
  panel rows.

The agent-visible public schema contains:

- identity: `datetime`, `symbol`;
- candidate predictors: `open`, `high`, `low`, `close`, `volume`,
  `quote_asset_volume`, `number_of_trades`,
  `taker_buy_base_asset_volume`, and `taker_buy_quote_asset_volume`;
- fitting metadata: `weight_std_dollar_vol`;
- supervised response: `target_horizon_1`.

`target_horizon_1` is intentionally anonymized. `weight_std_dollar_vol` is fitting
metadata rather than a predictor. Two additional columns, `tradable_return` and
`beta_10d_fwd_1`, are retained only by the trusted backtest and verifier. They
are never included in model fitting or prediction matrices.

The canonical public and hidden Parquet files are approximately 330 MB and
176 MB respectively. The agent image receives a compressed projection of the
public artifact with only identity, predictor, weight, and target columns.
Hidden data remain exclusively in the verifier environment. The dataset is
bundled with the task; no external download is required at runtime.

## Evaluation Strategy

The benchmark uses a continuous outcome reward rather than a binary pass
threshold. The primary score is annualized Sharpe on the fixed hidden 2024
after-cost strategy returns. The same value is emitted as `primary_score`,
`reward`, and `sharpe`. The verifier also reports hidden CAGR, maximum drawdown,
and Pearson information coefficient so that risk-adjusted performance can be
interpreted alongside predictive quality and downside behavior.

The portfolio backtest applies a one-basis-point linear execution fee and uses
525,600 periods per year for minute-level Sharpe annualization. The submitted
`max_gross_exposure` must be finite and strictly positive. Public and hidden
backtests use the same task-owned portfolio construction and accounting code.
The absence of a binary cutoff is intentional: the scalar reward supports
ranking, reinforcement learning, and comparison across research policies.

Before scoring, the verifier enforces the following safeguards:

1. **Submission integrity.** Exactly one publicly accepted bundle must exist.
   Source and artifact hashes must match the manifest, and bundle paths may not
   escape the submission directory.
2. **Estimator family.** `train_model` must return either an exact
   `sklearn.linear_model.ElasticNet` or an exact sklearn `Pipeline` whose final
   step is an exact `ElasticNet`. Subclasses and other final estimators are
   rejected.
3. **Executable contract.** Submitted source is size- and import-restricted.
   The estimator must be fitted, expose unique ordered string
   `feature_names_in_`, serialize with Joblib, reload in a fresh process, and
   return finite predictions in the original row order.
4. **Full-public refit.** Hidden scoring uses a new estimator fitted by calling
   the submitted `train_model` source once on the complete public 2022–2023
   panel. The estimator fitted during public experimentation is not scored.
5. **Causal audit.** Three hidden future-suffix probes materially perturb later
   feature rows. Predictions for earlier rows must remain invariant within
   strict numerical tolerances.
6. **Hidden isolation.** The model worker receives feature matrices but never
   hidden targets, tradable returns, market betas, or task-owned portfolio
   state.

A missing final submission receives `-100` for every reward field. Contract
violations are rejected before hidden economic metrics are exposed.

## Complexity

The computational primitives are standard, but the research problem is
difficult because useful minute-level signals are weak and easy to overfit.
Every candidate must survive chronological separation, cross-symbol analysis,
causal batch execution, regularization, portfolio normalization, turnover, and
costs. Fixing the final estimator to Elastic Net removes model-family search as
an escape hatch: the agent's advantage must come from better hypotheses,
features, controls, and candidate selection.

The public panel contains more than 4.2 million rows, and accepted training and
backtesting calls share a 100-attempt budget. A strong policy must allocate that
budget among data diagnostics, feature ablations, regularization comparisons,
chronological confirmation, and exposure experiments while asynchronous jobs
are running.

Observed task runs demonstrate a broad outcome range. The current reference
strategy receives hidden Sharpe `0.3723068959` with hidden Pearson IC
`0.0196662380`. Recorded end-to-end and candidate-validation artifacts include
missing or invalid submissions at `-100`, valid strategies with negative Sharpe,
degenerate historical baselines at `0`, and positive candidates between
approximately `0.15` and `0.39`. These are continuous outcomes rather than
pass/fail labels.

A representative Harbor run completed in 20 minutes 53 seconds end to end:
14 seconds for environment setup, 28 seconds for agent setup, 16 minutes
40 seconds for agent execution, and 3 minutes 11 seconds for verification. The
configured limits leave room for substantially deeper research than the
reference policy performs.

## References & Resources

The sample is self-contained. Its implementation contracts and supporting
evidence are documented in:

- `instruction.md` — agent-facing objective, data split, MCP workflow, and
  full-public refit disclosure;
- `environment/agent/context/research.md` — causal research and chronological
  validation guidance;
- `environment/agent/context/submission.md` — exact estimator, source,
  serialization, and prediction contract;
- `environment/agent/context/backtesting.md` — public candidate-comparison and
  strategy-selection guidance;
- `DATA_PROVENANCE.md` — fixed artifact hashes, source universe,
  anonymization, split construction, and license record;
- `environment/mcp-server/task_config.yaml` — task-owned data, backtest, cost,
  execution, reward, and budget configuration;
- `environment/mcp-server/feature_engineering/evaluation/verifier.py` and
  `scoring/official.py` — independent verification and hidden scoring flow;
- `solution/solve.py` — reference MCP-only agent;
- `environment/mcp-server/tests/` — causal, portfolio, runtime, release,
  end-to-end, and verifier behavior contracts.

The complete local Python verification command is:

```sh
uv run --project environment/mcp-server test
```

## Additional Information

This document describes the OpenAI-facing sample as an outcome-scored research
environment rather than a binary programming exercise. Its hidden Sharpe reward
is intended for continuous evaluation, policy comparison, and reinforcement
learning. The fixed Elastic Net family is an experimental control that focuses
the task on feature engineering and quantitative-research decisions.

## Author Information

**Contributing organization:** EdotEnv  
**Contribution:** Task design, implementation, data packaging, verifier,
reference policy, and calibration.
