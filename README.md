# Autoresearch Lab

A framework for running automated AI research loops where an agent iteratively improves code against a measurable benchmark. Currently powered by [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

You define a **pipeline** (the code being optimized), a **backend** implemented in Python (how to evaluate it), and **data** (what to evaluate against). The framework handles sandboxing, orchestration, git commits/reverts, and stopping conditions.

## Background

This project is inspired by Andrej Karpathy's [autoresearch pattern](https://github.com/karpathy/autoresearch) — the idea that an AI agent can autonomously run a research loop of "change → evaluate → keep/discard" against a benchmark, accumulating improvements over time.

Autoresearch Lab implements the pattern in a somewhat generic way:

- **Language and domain agnostic.** It treats the pipeline (the code being optimized) as a black box — it can be in written in any language and run in any environment, as long as it can be evaluated resulting in a single score to optimize. The tradeoff is that you have to implement an evaluation backend (in Python) as a bridge to your code.
- **Sandboxed by default.** The agent runs inside a Docker container with only the pipeline code mounted as writable, protecting the host from most rogue agent behavior. An orchestrator on the host manages git commits and reverts, so the agent loop can't corrupt your repo.
- **Integrates into existing git repos.** You can create a "lab" inside an existing git repo to persist the research loops's configuration and state. This makes it easy to stop and continue the loop, to collaborate on it with others and to keep developing a piece of code through a combination of autoresearch and human input.
- **Host service support.** Some pipelines can't run inside a Docker container (e.g. mobile code that needs an emulator or device). The framework can manage host-side services and forward ports into the sandbox, letting the agent's code run on real hardware while still being orchestrated.

### Warning: Sandbox limitations

The agent runs in a Docker container, which provides process-level isolation but is **not a true security sandbox**. A Docker container is not suitable for running untrusted or adversarial code. It is therefore risky to run Autoresearch Lab on a random machine. If you need stronger isolation, run Autoresearch Lab itself inside a VM.

No matter how good your isolation, the sandbox has network access because the agent needs to reach the API of the AI provider. This means the agent can make arbitrary HTTP requests, is vulnerable to prompt injection and may exfiltrate your pipeline code and data. **Do not include any secrets in your pipeline code and data.**

## Getting started

### Prerequisites

- **Docker** running (the agent runs in a sandboxed container)
- **Git** repository (the orchestrator commits/reverts pipeline changes)
- **Anthropic API key** set as environment variable `ANTHROPIC_API_KEY` (or use `--use-oauth-osx` if logged in via `claude login`)

### Quick start

This assumes you have [uv](https://docs.astral.sh/uv/#installation) for your Python dependency needs (but any other Python packaging solution will do).

```bash
# Initialize a lab in your project
cd my-project
uv init
uv add autoresearch-lab
uv run arl init --name "my-lab"

# Edit the generated files:
#   lab.toml      — configure backend and pipeline location
#   backend.py    — implement evaluation logic
#   AGENT.md      — write agent instructions

# Run the research loop
uv run arl run --data ./my-data --max-iterations 20

# Pass extra arguments to Claude Code after --
uv run arl run --data ./my-data -- --effort high
```

## Usage

### How it works

```
Host (arl run)                      Docker Container (Claude Code agent)
├─ Start host service (if any)       ├─ Read AGENT.md
├─ Launch container                  ├─ arl diagnose → understand failures
├─ Poll for verdict.json             ├─ Modify pipeline/ code
│   ◄── {"action": "keep", ...}  ◄──├─ arl eval → get metrics
├─ git commit or revert              ├─ Write verdict.json
├─ Delete verdict.json               ├─ Wait for deletion
├─ Check stopping conditions         ├─ Repeat
└─ Repeat                            └─ Repeat
```

The agent runs inside a sandboxed Docker container with restricted filesystem access. It can only modify the pipeline code and signal verdicts. Git operations happen on the host.

### CLI commands

| Command        | Description                                   |
| -------------- | --------------------------------------------- |
| `arl init`     | Initialize a new lab in the current directory |
| `arl run`      | Start the autonomous research loop            |
| `arl eval`     | Evaluate pipeline (prints JSON metrics)       |
| `arl diagnose` | Per-sample error analysis, worst first        |
| `arl results`  | Print experiment history from results.tsv     |

### Configuration

Each lab is configured via a `lab.toml` file:

```toml
[lab]
name = "my-lab"
pipeline_dir = "pipeline"           # The only code the agent can modify (mounted read-write)
agent_instructions = "AGENT.md"     # What the agent reads on startup
results_file = "results.tsv"        # Experiment log

[backend]
module = "backend.py"               # Python file implementing EvalBackend
class = "MyBackend"                 # Class name within that file

# Optional: host-side service started before the container
[backend.host_service]
command = "python daemon.py --port 9100"
port = 9100                         # Forwarded into the container

# Optional: custom Dockerfile (must use FROM arl-agent-base)
[sandbox]
dockerfile = "Dockerfile"
```

### Writing a backend

Implement `EvalBackend` from `autoresearch_lab.harness.backend`:

```python
from pathlib import Path
from autoresearch_lab.harness.backend import EvalBackend, EvalResult, SampleResult

class MyBackend(EvalBackend):
    def evaluate(self, pipeline_dir: Path, data_dir: Path,
                 sample_ids: list[str] | None = None) -> EvalResult:
        # Run pipeline, compare against ground truth, compute your score.
        # The score is the single number the framework tracks (lower is better).
        # Extra metrics are logged but the framework only uses the score.
        #
        # If sample_ids is provided, you can optionally restrict evaluation
        # to just those samples to speed up `arl diagnose --sample <id>`.
        return EvalResult(
            score=0.042,
            metrics={"my_metric": 0.042, "latency_ms": 150.0},
            sample_results=[
                SampleResult(sample_id="img_001", score=0.03),
                SampleResult(sample_id="img_002", score=0.05),
            ],
        )
```

The framework is metric-agnostic — your backend defines what the score means (CER, loss, error rate, etc.) and how to compute it. Additional metrics in the `metrics` dict are logged to `results.tsv` for reference but the framework only tracks `score` for keep/discard decisions and stopping conditions.

The backend runs inside the Docker container by default. If evaluation requires host resources (e.g. an Android emulator), declare a `[backend.host_service]` in `lab.toml` — the framework starts it on the host, waits for the port to be reachable, and forwards it into the container.

### Data

The `--data` argument passed to `arl run`, `arl eval`, and `arl diagnose` is a path to a directory containing whatever your backend needs to evaluate the pipeline — ground truth files, test inputs, reference images, benchmark configs, etc. The framework treats it as an opaque, read-only directory.

Each item in the data directory that produces a score is a **sample**. The backend returns per-sample results via `SampleResult`, which `arl diagnose` uses to show the worst-performing samples first. The structure of samples is entirely up to you — a sample could be a single file, a subdirectory, or an entry in a manifest file.

The data directory is not configured in `lab.toml` because it's common to evaluate against different datasets (e.g. a small fast set during development vs. a full set for final evaluation), and datasets they may not be a part of the code's git repo.

### Custom container

By default the agent runs in a base container with Python 3.12, Node 22, uv, and Claude Code. If your lab needs extra dependencies (system packages, runtimes, tools), create a `Dockerfile` that extends the base:

```dockerfile
FROM arl-agent-base

USER root
RUN apt-get update && apt-get install -y my-package
USER agent
```

Then point to it in `lab.toml`:

```toml
[sandbox]
dockerfile = "Dockerfile"
```

### Stopping conditions

`arl run` stops when any condition is met:

- `--max-iterations` — maximum number of experiments (default: 50)
- `--max-hours` — maximum session duration in hours (default: 8)
- `--max-cost` — maximum estimated API spend in USD (default: unlimited)
- `--target-score` — stop when score reaches this value (default: disabled)
- `--plateau-threshold` — consecutive discards without improvement (default: 10)
- `--max-restarts` — container crash or timeout restarts before stopping (default: 3)

## Development

```bash
uv sync              # Install dependencies (includes dev tools)
uv run pytest        # Run tests
uv run ruff check    # Lint
uv run ruff format   # Format code
uv run pyright       # Type check
```

## License

Autoresearch Lab is distributed under an [MIT license](LICENSE.txt).
