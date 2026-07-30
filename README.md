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
read-only model-worker container. Submitted source import, fitting, Joblib load,
and `.predict()` all happen in that worker; hidden labels never do.

## Run locally

```sh
uvx harbor run -p . -a oracle -e docker -n 1
```

Run with Harbor's Kimi adapter by selecting `kimi-cli` and a provider/model:

```sh
ALLOWED_TOOLS='["kimi_cli.tools.agent:Agent","kimi_cli.tools.todo:SetTodoList","kimi_cli.tools.shell:Shell","kimi_cli.tools.background:TaskList","kimi_cli.tools.background:TaskOutput","kimi_cli.tools.background:TaskStop","kimi_cli.tools.file:ReadFile","kimi_cli.tools.file:Glob","kimi_cli.tools.file:Grep","kimi_cli.tools.file:WriteFile","kimi_cli.tools.file:StrReplaceFile"]'
uvx harbor run -p . -a kimi-cli -m <provider/model> \
  --agent-kwarg version=1.49.0 \
  --agent-kwarg "allowed_tools=$ALLOWED_TOOLS" \
  --agent-kwarg max_context_size=<tokens> \
  -e docker -n 1
```

Harbor automatically registers the task's streamable-HTTP MCP server. Kimi's
context window can be set with the adapter's `max_context_size` argument shown
above; otherwise Harbor derives it from model metadata.

The `allowed_tools` argument requires a Harbor build whose `kimi-cli` adapter
supports that structured option. Harbor converts it to an ephemeral Kimi agent
file inside the running agent container; neither task image contains Kimi
configuration. Kimi's default subagent definitions remain inherited. The root
agent receives the listed built-in tools plus the task's five MCP tools;
subagents retain their native tool policies and do not receive the task MCP
configuration.

The verifier writes numeric `reward.json` fields for `reward`, `sharpe`, `cagr`,
`max_drawdown`, and `pearson_ic`; Sharpe is the scalar reward.
