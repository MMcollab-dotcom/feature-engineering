# Feature Engineering

A fixed Harbor feature-engineering task on public cryptocurrency
minute data, with hidden scoring after submission.

The agent phase has two containers:

- `main`: the public Parquet, three context Markdown files, and Python libraries
- `mcp-server`: asynchronous train, backtest, and submission tools

Both images derive their public data from one canonical Parquet snapshot. The
main image materializes only the agent-visible columns as compressed Parquet;
the two scoring columns remain available only to the MCP runtime.

Harbor collects `/app/submission` directly from the MCP sidecar, tears the agent
environment down, and starts a separate verifier. The verifier owns hidden data
and sends only feature matrices and the submitted bundle to a networkless,
read-only model-worker container. The verifier imports the submitted source,
refits it once on all public 2022–2023 rows, serializes and reloads that new
estimator, and calls `.predict()` on hidden 2024 features. Hidden labels never
enter the worker; the estimator artifact fitted during public research is not
used for the hidden score.

## Run with Harbor

From the repository root:

```sh
git lfs pull
uvx harbor==0.20.0 run -p . -a codex -m openai/gpt-5.6-luna \
  --agent-kwarg reasoning_effort=medium \
  -e docker -n 1
```
