"""arl — CLI for Autoresearch Lab.

All commands are implicitly scoped to the nearest lab.toml.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import click

from autoresearch_lab.config import LAB_CONFIG_FILENAME, BackendConfig, LabConfig
from autoresearch_lab.harness.loader import load_backend
from autoresearch_lab.harness.results import read_results
from autoresearch_lab.sandbox.orchestrator import (
    CONTAINER_PIPELINE_DIR,
    RunConfig,
    run_session,
    start_host_service,
)
from autoresearch_lab.template_loader import render_template


def _echo_success(msg: str) -> None:
    click.echo(click.style(msg, fg="green"))


def _echo_warn(msg: str) -> None:
    click.echo(click.style(msg, fg="yellow"))


def _echo_fail(msg: str) -> None:
    click.echo(click.style(msg, fg="red"))


def _find_lab() -> tuple[LabConfig, Path]:
    """Find and load the nearest lab.toml."""
    lab_root = LabConfig.find_lab_root()
    config = LabConfig.load(lab_root)
    return config, lab_root


def _render_template(template_name: str, **kwargs: str) -> str:
    """Read a template file and substitute placeholders."""
    return render_template(template_name, **kwargs)


@contextmanager
def _maybe_host_service(
    config: LabConfig,
    lab_root: Path,
    data_dir: Path,
    pipeline_dir: Path,
):
    """Start the host service if configured and not already
    available (i.e. inside the container).

    When ARL_HOST_SERVICE_URL is already set (inside the
    container), this is a no-op. When running locally, starts
    the service, sets the env var, and cleans up on exit.
    """
    host_service = config.backend.host_service
    if not host_service or os.environ.get("ARL_HOST_SERVICE_URL"):
        yield
        return

    proc = start_host_service(host_service, lab_root, data_dir, pipeline_dir)
    os.environ["ARL_HOST_SERVICE_URL"] = f"http://localhost:{host_service.port}"
    try:
        yield
    finally:
        os.environ.pop("ARL_HOST_SERVICE_URL", None)
        proc.terminate()
        proc.wait(timeout=5)


@click.group()
@click.version_option(package_name="autoresearch-lab")
def cli():
    """Autoresearch Lab — automated AI research loops."""


@cli.command()
@click.option("--name", prompt="Lab name", help="Name for this lab")
def init(name: str):
    """Initialize a new lab in the current directory."""
    lab_root = Path.cwd()
    config_path = lab_root / LAB_CONFIG_FILENAME

    if config_path.exists():
        click.echo(f"Error: {config_path} already exists", err=True)
        sys.exit(1)

    config_path.write_text(_render_template("lab.toml", name=name))

    backend_path = lab_root / BackendConfig.module
    if not backend_path.exists():
        backend_path.write_text(_render_template("backend.py"))

    agent_md = lab_root / LabConfig.agent_instructions
    if not agent_md.exists():
        agent_md.write_text(
            _render_template(
                "AGENT.md",
                name=name,
                pipeline_dir=CONTAINER_PIPELINE_DIR,
            )
        )

    (lab_root / LabConfig.results_file).touch()

    click.echo(f"Initialized lab '{name}' in {lab_root}")
    click.echo(f"  Set pipeline_dir in {config_path.name} to point at your code")
    click.echo("  Write agent instructions in AGENT.md")
    click.echo(f"  Implement evaluation logic in {BackendConfig.module}")


def _preflight_checks(config: LabConfig, lab_root: Path) -> None:
    """Run pre-flight checks. Exits with error if required checks fail."""
    failed = False

    def _check(label: str, ok: bool, required: bool = True) -> None:
        nonlocal failed
        if ok:
            _echo_success(f"  {label}: ok")
        elif required:
            failed = True
            _echo_fail(f"  {label}: FAILED")
        else:
            _echo_warn(f"  {label}: not set")

    click.echo("Pre-flight checks:")
    result = subprocess.run(["docker", "info"], capture_output=True, text=True)
    _check("Docker", result.returncode == 0)
    _check(
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY" in os.environ,
        required=False,
    )
    _check(
        f"Backend ({config.backend.module})",
        (lab_root / config.backend.module).exists(),
    )
    _check(
        "Pipeline dir",
        (lab_root / config.pipeline_dir).exists(),
    )
    _check(
        "Agent instructions",
        (lab_root / config.agent_instructions).exists(),
    )

    if failed:
        _echo_fail("\nPre-flight checks failed. Aborting.")
        sys.exit(1)
    click.echo()


@cli.command()
@click.option("--data", required=True, help="Path to read-only data directory")
@click.option(
    "--max-iterations",
    default=RunConfig.max_iterations,
    help="Max experiments",
)
@click.option(
    "--max-hours",
    default=RunConfig.max_hours,
    help="Max session hours",
)
@click.option(
    "--max-cost",
    default=RunConfig.max_cost,
    help="Max USD spend (0=unlimited)",
)
@click.option(
    "--target-score",
    default=RunConfig.target_score,
    help="Stop when score reaches this value (0=disabled)",
)
@click.option(
    "--max-restarts",
    default=RunConfig.max_restarts,
    help="Max container restarts before stopping",
)
@click.option(
    "--iteration-timeout",
    default=RunConfig.iteration_timeout,
    help="Seconds before restarting a stuck iteration (0=disabled)",
)
@click.option("--model", default=RunConfig.model, help="Claude model")
@click.option("--use-oauth-osx", is_flag=True, help="Use OAuth from Keychain")
@click.option("--prompt", default=RunConfig.prompt, help="Additional agent instruction")
@click.option("--dry-run", is_flag=True, help="Print config and exit")
@click.argument("claude_args", nargs=-1, type=click.UNPROCESSED)
def run(claude_args, **kwargs):
    """Start the autonomous research loop.

    Extra arguments after -- are passed through to Claude Code.
    """
    config, lab_root = _find_lab()

    image_tag = f"arl-agent-{config.safe_name}"
    run_cfg = RunConfig(**kwargs, docker_image=image_tag, claude_args=claude_args)

    if run_cfg.dry_run:
        click.echo(f"Lab:       {config.name}")
        click.echo(f"Root:      {lab_root}")
        click.echo(f"Pipeline:  {lab_root / config.pipeline_dir}")
        click.echo(f"Backend:   {config.backend.module}:{config.backend.cls}")
        click.echo(f"Data:      {run_cfg.data}")
        click.echo(f"Model:     {run_cfg.model}")
        return

    _preflight_checks(config, lab_root)

    # Build base image
    base_dockerfile = Path(__file__).parent / "sandbox" / "Dockerfile"

    # If running from a source checkout, install from local source.
    # Otherwise the Dockerfile defaults to installing from PyPI.
    package_root = Path(__file__).parent.parent.parent
    build_args = []
    if (package_root / "pyproject.toml").exists():
        build_args = [
            "--build-arg",
            "ARL_INSTALL_SPEC=/tmp/autoresearch-lab",
        ]
        # We need the source in the build context for COPY
        context = str(package_root)
    else:
        from importlib.metadata import version

        arl_version = version("autoresearch-lab")
        build_args = [
            "--build-arg",
            f"ARL_INSTALL_SPEC=autoresearch-lab=={arl_version}",
        ]
        context = str(base_dockerfile.parent)

    click.echo("Building base agent container...")
    result = subprocess.run(
        [
            "docker",
            "build",
            "-t",
            "arl-agent-base",
            "-f",
            str(base_dockerfile),
            *build_args,
            context,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Docker build failed:\n{result.stderr}", err=True)
        sys.exit(1)

    # Build custom image if lab provides a Dockerfile
    if config.sandbox.dockerfile:
        custom_dockerfile = (lab_root / config.sandbox.dockerfile).resolve()
        if not custom_dockerfile.exists():
            click.echo(
                f"Error: sandbox dockerfile not found: {custom_dockerfile}",
                err=True,
            )
            sys.exit(1)
        build_context = config.sandbox.build_context or custom_dockerfile.parent
        click.echo(f"Building custom agent container ({image_tag})...")
        result = subprocess.run(
            [
                "docker",
                "build",
                "-t",
                image_tag,
                "-f",
                str(custom_dockerfile),
                str(build_context),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            click.echo(f"Docker build failed:\n{result.stderr}", err=True)
            sys.exit(1)
    else:
        subprocess.run(
            ["docker", "tag", "arl-agent-base", image_tag],
            capture_output=True,
        )

    run_session(run_cfg, config, lab_root)


LAST_EVAL_FILENAME = ".last_eval.json"


@cli.command()
@click.option("--data", required=True, help="Path to read-only data directory")
def eval(data):
    """Run pipeline evaluation. Prints metrics as JSON.

    The result is cached so that `arl verdict` can reference it
    without re-running the evaluation.
    """
    config, lab_root = _find_lab()

    pipeline_dir = (lab_root / config.pipeline_dir).resolve()
    data_dir = Path(data).resolve()

    if not pipeline_dir.exists():
        click.echo(f"Error: pipeline dir not found: {pipeline_dir}", err=True)
        sys.exit(1)
    if not data_dir.exists():
        click.echo(f"Error: data dir not found: {data_dir}", err=True)
        sys.exit(1)

    with _maybe_host_service(config, lab_root, data_dir, pipeline_dir):
        backend = load_backend(lab_root, config.backend)
        backend.setup()
        try:
            result = backend.evaluate(pipeline_dir, data_dir)
        finally:
            backend.teardown()

    output = {"score": result.score, **result.metrics}
    output["num_samples"] = len(result.sample_results)
    click.echo(json.dumps(output, indent=2))

    # Cache for `arl verdict`
    cache = {"score": result.score, "metrics": result.metrics}
    cache_path = lab_root / LAST_EVAL_FILENAME
    cache_path.write_text(json.dumps(cache))


@cli.command()
@click.option(
    "--action",
    type=click.Choice(["keep", "discard"]),
    required=True,
    help="Keep or discard the current experiment",
)
@click.option("--verdict-path", required=True, help="Path to write verdict JSON")
@click.option("--experiment-id", default=None, help="Experiment ID")
@click.option("--notes", default="", help="Description of change")
def verdict(action, verdict_path, experiment_id, notes):
    """Write a verdict using the score and metrics from the last eval.

    This avoids re-running the evaluation and guarantees the verdict
    matches the eval output exactly.
    """
    _config, lab_root = _find_lab()
    cache_path = lab_root / LAST_EVAL_FILENAME

    if not cache_path.exists():
        click.echo(
            "Error: no cached eval result. Run `arl eval` first.",
            err=True,
        )
        sys.exit(1)

    cached = json.loads(cache_path.read_text())

    verdict_data = {
        "action": action,
        "score": cached["score"],
        "metrics": cached["metrics"],
        "notes": notes,
    }
    if experiment_id:
        verdict_data["experiment_id"] = experiment_id

    out = Path(verdict_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict_data))
    click.echo(json.dumps(verdict_data, indent=2))


@cli.command()
@click.option("--data", required=True, help="Path to read-only data directory")
@click.option("--top", type=int, default=None, help="Show only N worst samples")
@click.option("--sample", default=None, help="Show only a specific sample by ID")
def diagnose(data, top, sample):
    """Per-sample error analysis, worst first."""
    config, lab_root = _find_lab()

    pipeline_dir = (lab_root / config.pipeline_dir).resolve()
    data_dir = Path(data).resolve()

    with _maybe_host_service(config, lab_root, data_dir, pipeline_dir):
        backend = load_backend(lab_root, config.backend)
        backend.setup()
        try:
            sample_ids = [sample] if sample else None
            result = backend.evaluate(pipeline_dir, data_dir, sample_ids)
        finally:
            backend.teardown()

    if sample:
        matched = [s for s in result.sample_results if s.sample_id == sample]
        if not matched:
            click.echo(f"Sample '{sample}' not found", err=True)
            sys.exit(1)
        output = {
            "sample_id": matched[0].sample_id,
            "score": matched[0].score,
            "error": matched[0].error,
            **matched[0].extra,
        }
        click.echo(json.dumps(output, indent=2))
        return

    # Sort by score, worst first
    sorted_samples = sorted(
        result.sample_results,
        key=lambda s: s.score,
        reverse=True,
    )

    if top:
        sorted_samples = sorted_samples[:top]

    output = {
        "score": result.score,
        "metrics": result.metrics,
        "per_sample": [
            {
                "sample_id": s.sample_id,
                "score": s.score,
                "error": s.error,
                **s.extra,
            }
            for s in sorted_samples
        ],
    }

    click.echo(json.dumps(output, indent=2))


@cli.command()
@click.option("--best", is_flag=True, help="Show only the best result")
@click.option("--last", type=int, default=None, help="Show last N results")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
def results(best, last, fmt):
    """Print experiment history."""
    config, lab_root = _find_lab()

    results_path = lab_root / config.results_file
    rows = read_results(results_path)

    if not rows:
        click.echo("No results yet.")
        return

    if best:
        rows = [min(rows, key=lambda r: float(r.get("score", "999")))]

    if last:
        rows = rows[-last:]

    if fmt == "json":
        click.echo(json.dumps(rows, indent=2))
    elif fmt == "csv":
        if rows:
            click.echo("\t".join(rows[0].keys()))
            for row in rows:
                click.echo("\t".join(row.values()))
    else:
        for row in rows:
            kept = (
                click.style("KEEP", fg="green")
                if row.get("kept") == "yes"
                else click.style("DISC", fg="yellow")
            )
            exp = row.get("experiment_id", "?")
            score = row.get("score", "?")
            notes = row.get("notes", "")
            # Parse extra metrics from JSON column
            try:
                extra_metrics = json.loads(row.get("metrics", "{}"))
            except json.JSONDecodeError:
                extra_metrics = {}
            metrics_str = "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in extra_metrics.items()
            )
            extra = f"  {metrics_str}" if metrics_str else ""
            click.echo(f"  {kept} {exp:>12s}  score={score}{extra}  {notes}")


def _parse_label_path(arg: str) -> tuple[str, str]:
    """Parse a ``label:path`` argument or infer the label.

    Supports explicit ``"My Label:/some/results.tsv"`` syntax.
    If no label prefix is given, tries to read the lab name from
    a sibling ``lab.toml``, falling back to the parent directory name.
    """
    # Only split on the first colon, but not if it looks like a
    # Windows drive letter (e.g. C:\\...) or the part before the
    # colon doesn't exist as a file.
    if ":" in arg:
        head, tail = arg.split(":", 1)
        if head and tail and not Path(arg).exists():
            return head, tail

    # Auto-detect label
    p = Path(arg)
    lab_toml = p.parent / LAB_CONFIG_FILENAME
    if lab_toml.exists():
        try:
            cfg = LabConfig.load(p.parent)
            if cfg.name:
                return cfg.name, arg
        except Exception:
            pass
    return p.parent.name, arg


@cli.command()
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output file path (e.g. progress.png). Displays interactively if omitted.",
)
@click.option(
    "--title",
    default=None,
    help="Custom chart title (default: 'Autoresearch Lab Progress').",
)
@click.option(
    "--ymin",
    default=None,
    type=float,
    help="Lower y-axis bound.",
)
@click.option(
    "--ymax",
    default=None,
    type=float,
    help="Upper y-axis bound (cap outliers).",
)
@click.option(
    "--ylabel",
    default=None,
    help="Custom y-axis label (default: 'Score (lower is better)').",
)
@click.option(
    "--xlabel",
    default=None,
    help="Custom x-axis label (default: 'Experiment #').",
)
@click.option(
    "--figsize",
    default=None,
    help="Figure size as WIDTHxHEIGHT in inches (default: 14x7).",
)
@click.option(
    "--no-labels",
    is_flag=True,
    help="Hide text annotations on kept experiments.",
)
@click.argument("extra_results", nargs=-1)
def plot(output, title, ymin, ymax, ylabel, xlabel, figsize, no_labels, extra_results):
    """Plot experiment progress chart from results.

    Optionally pass extra results.tsv file paths to compare
    multiple labs in the same chart. Use LABEL:PATH syntax
    to set a custom label (e.g. "iOS Lab:../ios/results.tsv").
    Otherwise the label is inferred from a sibling lab.toml
    or the parent directory name.

    Requires matplotlib: install with `pip install autoresearch-lab[plot]`
    """
    try:
        from autoresearch_lab.plot import plot_results
    except ImportError:
        click.echo(
            "Error: matplotlib is required for plotting.\n"
            "Install with: pip install 'autoresearch-lab[plot]'",
            err=True,
        )
        sys.exit(1)

    # Build list of (label, rows) to plot
    series: list[tuple[str, list[dict[str, str]]]] = []

    config, lab_root = _find_lab()
    results_path = lab_root / config.results_file
    rows = read_results(results_path)
    if rows:
        lab_label = config.name or lab_root.name
        series.append((lab_label, rows))

    for arg in extra_results:
        label, path_str = _parse_label_path(arg)
        p = Path(path_str)
        if not p.exists():
            click.echo(f"Error: file not found: {p}", err=True)
            sys.exit(1)
        extra_rows = read_results(p)
        if extra_rows:
            series.append((label, extra_rows))

    if not series:
        click.echo("No results yet.")
        return

    parsed_figsize = None
    if figsize:
        try:
            w, h = figsize.split("x")
            parsed_figsize = (float(w), float(h))
        except ValueError:
            click.echo(
                "Error: --figsize must be WIDTHxHEIGHT (e.g. 14x7)",
                err=True,
            )
            sys.exit(1)

    plot_results(
        series,
        output,
        title=title,
        ymin=ymin,
        ymax=ymax,
        ylabel=ylabel,
        xlabel=xlabel,
        figsize=parsed_figsize,
        show_labels=not no_labels,
    )
    if output:
        click.echo(f"Saved plot to {output}")
