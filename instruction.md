# Fixed feature-engineering task

Discover causal features, train compact sklearn models, compare them
on the public split, and submit one successfully backtested strategy.

Start by reading `context/research.md`, `context/submission.md`, and
`context/backtesting.md`. The only dataset available to you is
`data/public_train.csv`: 2022–2023 minute rows for four anonymized symbols. The
target is the opaque `target_horizon_1`. Treat `weight_std_dollar_vol` only as
sample-weight metadata, never as a predictor.

Use these MCP tools directly; their live schemas are authoritative:

- `train_model`
- `get_train_model_result`
- `backtest`
- `get_backtest_result`
- `submit_strategy`

Pass complete Python source to `train_model`. Training and backtesting are
asynchronous: start an operation, do other work, then query its getter until
`done` is true. One training and one backtest may overlap, but two operations of
the same type may not.

Submit the `strategy_id` from your chosen successful backtest only after both
operation slots are idle. Hidden 2024 scoring happens after the agent exits.
