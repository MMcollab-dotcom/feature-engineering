# Backtesting guide

Use a public backtest after a candidate has credible chronological prediction
evidence and again before final submission. A backtest is diagnostic, not a
substitute for target prediction.

Every official candidate comparison must use disjoint chronological windows:
call `train_model` on an earlier `train_filter`, then call `backtest` with a
`backtest_filter` whose first datetime is strictly later than the training end.
Do not omit both filters or backtest any row used to fit that model. Reuse the
same disjoint windows when comparing candidates, and confirm the preferred
candidate on a later slice when the remaining public period permits.

Compare candidates on consistent date windows and inspect after-cost Sharpe,
CAGR, maximum drawdown, turnover, prediction correlation, and error together.
Treat isolated high Sharpe with weak or unstable IC as likely overfit. Likewise,
good IC does not excuse consistently poor economic behavior. Use results to form
specific hypotheses about exposure scale, turnover, drawdown, or regime
sensitivity, then recheck any model change out of sample.

`backtest` starts asynchronously and returns a `backtest_id`. Query
`get_backtest_result` later. A successful terminal result contains the
`strategy_id` accepted by `submit_strategy`. `max_gross_exposure` must be finite
and positive; compare exposure choices deliberately rather than optimizing one
public period blindly.
