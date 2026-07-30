# Data Provenance and External-Provider Clearance

## Scope

This record covers the two canonical evaluation artifacts:

| Artifact | Coverage | SHA-256 |
|---|---|---|
| `environment/mcp-server/runtime_public.parquet` | Public forecast origins from 2022-01-01 through 2023-12-31 | `cc9c5f7f434886715b77189fc9790fea58cf71f3ec611917a59bd9b1634fd36a` |
| `tests/hidden_data/hidden.parquet` | Hidden forecast origins from 2024-01-01 through 2024-12-31 | `8e1dddee577a0e20fea9af99e6015112cccfd3b57a8cc5b71b4e650e55ebf5af` |

The hashes and exact forecast, execution, and realization endpoints are also
recorded in `environment/mcp-server/data_manifest.json`.

## Recoverable provenance

- Data type: historical cryptocurrency minute bars and supervised-learning
  fields.
- Source symbols: `BTCUSDT`, `DOGEUSDT`, `ETHUSDT`, and `SOLUSDT`.
- Upstream venue, API, or dataset: **not recorded**. Symbol names alone do not
  establish the venue.
- Acquisition date and acquiring party: **not recorded**.
- Upstream license or API terms in effect at acquisition: **not recorded**.
- Source location: intentionally omitted from the external distribution.

## Recorded transformations

1. Source symbol identities were replaced with `symbol_01` through
   `symbol_04`; no source-to-anonymized-symbol mapping is published.
2. Datetimes were replaced with synthetic UTC coordinates while preserving row
   order and elapsed durations. Calendar labels therefore do not identify
   historical events.
3. Forecast origins were split chronologically: 2022-2023 is public and 2024 is
   hidden. Each split retains later execution and realization rows required by
   the forecast horizon.
4. The distributed market fields are OHLCV, quote volume, trade count, taker-buy
   volumes, and `weight_std_dollar_vol`.
5. The supervised response is distributed as the opaque
   `target_horizon_1`. Its derivation is **not recorded**.
6. `tradable_return` and `beta_10d_fwd_1` are scoring-only columns. Their
   derivations are **not recorded**.
7. The agent image projects the canonical public artifact to agent-visible
   identity, feature, weight, and target columns and writes Zstandard-compressed
   Parquet. It does not receive either scoring-only column.

No other acquisition, cleaning, resampling, missing-data, outlier, or target
construction steps are documented in this repository.

## License in this repository

`environment/LICENSE` grants recipients permission from `ricefan-tech` to use,
copy, run, and modify the task code and bundled data solely for internal
evaluation, research, and demonstration. It prohibits redistribution, public
hosting, commercial deployment, and real-money trading without separate written
permission. It also says that it grants only rights held by the copyright holder
and leaves compliance with third-party terms to the recipient.

## OpenAI and Anthropic clearance

**Status: NOT CLEARED.** The repository does not contain enough evidence to
confirm that public data, derived rows, prompts containing those rows, or task
artifacts may be processed by OpenAI or Anthropic. In particular, the upstream
source terms and the copyright holder's provider-specific authorization are
absent.

Do not run the external-provider commands in `README.md`, upload a run to Harbor
Hub, or otherwise transmit the data or derived task artifacts until an
authorized data owner records all of the following outside or in a revision of
this file:

1. upstream venue/dataset and acquisition method;
2. acquisition date and the upstream terms that applied;
3. authority to sublicense or submit the data for third-party model processing;
4. separate approval for OpenAI and for Anthropic, including the applicable
   organization/project accounts and retention or model-training settings;
5. approver name, approval date, and a durable reference to the written
   authorization.

A private Harbor Hub upload is also an external transfer and is not cleared by
this record. A public upload additionally conflicts with the repository
license's public-hosting restriction unless separate written permission is
obtained.
