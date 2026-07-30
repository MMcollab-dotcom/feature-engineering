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

### Prerequisites

Use Harbor 0.20.0 and Git LFS. The `codex` and `claude-code` adapters are
bundled with that Harbor release; they are not separate Python packages.

```sh
git lfs install
git lfs pull
git lfs ls-files
uvx harbor==0.20.0 --version
```

`git lfs ls-files` must include
`environment/mcp-server/runtime_public.parquet` and
`tests/hidden_data/hidden.parquet`. Docker must have at least the task's
declared 4 CPUs, 8 GiB memory, and 32 GiB storage available.

The bundled data is licensed only for internal evaluation, research, and
demonstration. External-provider use is not yet cleared; read
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) before exporting an API key or
starting an OpenAI or Anthropic run.

### Local oracle

```sh
uvx harbor==0.20.0 run -p . -a oracle -e docker -n 1
```

Run with Harbor's Kimi adapter by selecting `kimi-cli` and a provider/model:

```sh
ALLOWED_TOOLS='["kimi_cli.tools.agent:Agent","kimi_cli.tools.todo:SetTodoList","kimi_cli.tools.shell:Shell","kimi_cli.tools.background:TaskList","kimi_cli.tools.background:TaskOutput","kimi_cli.tools.background:TaskStop","kimi_cli.tools.file:ReadFile","kimi_cli.tools.file:Glob","kimi_cli.tools.file:Grep","kimi_cli.tools.file:WriteFile","kimi_cli.tools.file:StrReplaceFile"]'
uvx harbor==0.20.0 run -p . -a kimi-cli -m <provider/model> \
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

### OpenAI Codex

After the data-use clearance in `DATA_PROVENANCE.md` is signed, run the
OpenAI Codex adapter with the tested Harbor and CLI versions:

```sh
export OPENAI_API_KEY=...
uvx harbor==0.20.0 run -p . -a codex -m gpt-5.4 \
  --agent-kwarg version=0.146.0 \
  -e docker -n 1
```

### Anthropic Claude Code

After the same clearance is signed, run the Anthropic adapter with:

```sh
export ANTHROPIC_API_KEY=...
uvx harbor==0.20.0 run -p . -a claude-code -m claude-sonnet-4-6 \
  --agent-kwarg version=2.1.220 \
  -e docker -n 1
```

The CLI pins are intentional: without `--agent-kwarg version=...`, Harbor
installs the latest agent CLI, making later runs harder to reproduce.

## Runtime and reference result

The configured limits are 30 minutes for environment build, four hours for the
agent, and one hour for verification. Cached local builds are much faster. A
representative Harbor run on 2026-07-29 took 20 minutes 53 seconds end to end:
14 seconds for environment setup, 28 seconds for agent setup, 16 minutes
40 seconds for agent execution, and 3 minutes 11 seconds for verification.

The retained release-audit reference strategy produced public Sharpe `1.56`
and hidden reward/Sharpe `0.1739`. A separate Kimi/OpenAI agent run produced
hidden reward/Sharpe `0.3874`. These are reproducibility landmarks, not a pass
threshold; compare scores only on the same task commit.

## Troubleshooting

- **Parquet files are tiny text pointers or missing:** run `git lfs pull`, then
  confirm both hashes with `git lfs ls-files`. Do not build until both objects
  are present.
- **Authentication fails:** export the provider key in the shell that launches
  Harbor. Do not add keys to task files, Dockerfiles, or committed `.env` files.
- **The agent setup changes between runs:** retain the Harbor and agent CLI
  version pins shown above.
- **Docker reports insufficient space:** inspect usage with `docker system df`
  and remove only disposable, unrelated build artifacts before retrying. The
  task requests 32 GiB of environment storage.
- **The MCP server never becomes healthy:** inspect the `mcp-server` container
  logs and verify port `8000` is reachable inside the Compose network.
- **Verification fails after the agent exits:** inspect the trial's
  `verifier/test-stdout.txt`, `verifier/metrics.json`, and
  `verifier/reward.json`. Submitted artifacts are collected from the
  `mcp-server` service, not the agent container.

The verifier writes numeric `reward.json` fields for `reward`, `sharpe`, `cagr`,
`max_drawdown`, and `pearson_ic`; Sharpe is the scalar reward.
