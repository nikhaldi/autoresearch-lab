"""Host-side orchestrator for autonomous research sessions.

Manages the research loop from the host:
  1. Optionally starts a host-side service (declared in lab.toml)
  2. Launches Claude Code inside a Docker container
  3. Watches for verdict.json signals from the container
  4. Performs git commit/revert based on the verdict
  5. Enforces stopping conditions (max iterations, time, plateau)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import click

from autoresearch_lab.config import HostServiceConfig, LabConfig
from autoresearch_lab.harness.results import append_result, read_results
from autoresearch_lab.net import is_port_open
from autoresearch_lab.sandbox.stream_formatter import CostTracker, start_stream_thread

VERDICT_FILENAME = "verdict.json"

# Container-side mount paths — these define the contract between the
# orchestrator and the agent. AGENT.md templates reference these paths.
CONTAINER_AGENT_MD = "/workspace/AGENT.md"
CONTAINER_DATA_DIR = "/workspace/data"
CONTAINER_PIPELINE_DIR = "/workspace/pipeline"
CONTAINER_RESULTS_FILE = "/workspace/results.tsv"
CONTAINER_VERDICTS_DIR = "/workspace/verdicts"
CONTAINER_VERDICT_PATH = f"{CONTAINER_VERDICTS_DIR}/{VERDICT_FILENAME}"
CONTAINER_LAB_TOML = "/workspace/lab.toml"
CONTAINER_BACKEND_DIR = "/workspace/backend"


@dataclass
class RunConfig:
    """Runtime configuration for arl run (from CLI flags)."""

    max_iterations: int = 50
    max_hours: float = 8.0
    plateau_threshold: int = 10
    target_score: float = 0.0
    eval_timeout: int = 300
    iteration_timeout: int = 900
    max_restarts: int = 3
    max_cost: float = 0.0
    model: str = "claude-opus-4-6"
    use_oauth_osx: bool = False
    prompt: str = ""
    data: str = ""
    dry_run: bool = False
    docker_image: str = "arl-agent"


@dataclass
class SessionState:
    start_time: float = 0.0
    iteration: int = 0
    consecutive_discards: int = 0
    best_score: float = 1.0
    restarts: int = 0
    last_verdict_time: float = 0.0


def check_stop_conditions(
    run_config: RunConfig, state: SessionState, cost_usd: float = 0.0
) -> str | None:
    """Return a reason to stop, or None to continue."""
    if state.iteration >= run_config.max_iterations:
        return f"Reached max iterations ({run_config.max_iterations})"

    elapsed_hours = (time.time() - state.start_time) / 3600
    if elapsed_hours >= run_config.max_hours:
        return f"Reached max time ({run_config.max_hours}h)"

    if state.consecutive_discards >= run_config.plateau_threshold:
        return (
            f"Plateau: {run_config.plateau_threshold} consecutive "
            f"discards without improvement"
        )

    if run_config.target_score > 0 and state.best_score <= run_config.target_score:
        return f"Reached target score ({run_config.target_score})"

    if run_config.max_cost > 0 and cost_usd >= run_config.max_cost:
        return f"Reached cost limit (${run_config.max_cost:.2f}, spent ${cost_usd:.2f})"

    return None


def git_commit(pipeline_dir: Path, message: str, cwd: Path) -> str:
    """Commit pipeline changes, return the commit SHA."""
    result = subprocess.run(
        ["git", "add", str(pipeline_dir)],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        print(f"  git add failed: {result.stderr}", file=sys.stderr)
        return "unknown"

    result = subprocess.run(
        ["git", "commit", "-m", message],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        print(f"  git commit failed: {result.stderr}", file=sys.stderr)
        return "unknown"

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return result.stdout.strip()


def git_amend_with_results(results_tsv: Path, cwd: Path) -> None:
    """Amend the last commit to include updated results.tsv."""
    subprocess.run(
        ["git", "add", str(results_tsv)],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    subprocess.run(
        ["git", "commit", "--amend", "--no-edit"],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def git_revert(path: Path, cwd: Path) -> None:
    """Revert a path to the last committed state."""
    subprocess.run(
        ["git", "checkout", "--", str(path)],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def start_host_service(
    service_config: HostServiceConfig,
    lab_root: Path,
    data_dir: Path,
    pipeline_dir: Path,
) -> subprocess.Popen:
    """Start the host-side service and wait until its port is accepting connections.

    Sets ARL_DATA_DIR and ARL_PIPELINE_DIR in the service's environment
    so the command can reference the actual paths.
    """
    env = {
        **os.environ,
        "ARL_DATA_DIR": str(data_dir),
        "ARL_PIPELINE_DIR": str(pipeline_dir),
    }
    proc = subprocess.Popen(
        service_config.command,
        shell=True,
        cwd=lab_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for _ in range(30):
        if is_port_open(service_config.port):
            return proc
        time.sleep(1)

    proc.terminate()
    raise RuntimeError(
        f"Host service on port {service_config.port} not reachable after 30s"
    )


def _get_oauth_token() -> str:
    """Extract OAuth token from macOS Keychain."""
    result = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-s",
            "Claude Code-credentials",
            "-w",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            "Error: Could not read OAuth token from Keychain. "
            "Run 'claude login' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    creds = json.loads(result.stdout.strip())
    token = creds.get("claudeAiOauth", {}).get("accessToken")
    if not token:
        print(
            "Error: No OAuth access token found in Keychain.",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def start_container(
    run_config: RunConfig,
    lab_config: LabConfig,
    lab_root: Path,
    verdict_path: Path,
    cost_tracker: CostTracker,
) -> subprocess.Popen:
    """Start the Docker container with Claude Code."""
    if run_config.use_oauth_osx:
        auth_token = _get_oauth_token()
        auth_env = ["-e", f"CLAUDE_CODE_OAUTH_TOKEN={auth_token}"]
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print(
                "Error: ANTHROPIC_API_KEY not set. Set it or use --use-oauth-osx.",
                file=sys.stderr,
            )
            sys.exit(1)
        auth_env = ["-e", f"ANTHROPIC_API_KEY={api_key}"]

    pipeline_dir = (lab_root / lab_config.pipeline_dir).resolve()
    results_file = (lab_root / lab_config.results_file).resolve()
    agent_instructions = (lab_root / lab_config.agent_instructions).resolve()
    backend_dir = (lab_root / Path(lab_config.backend.module).parent).resolve()
    data_dir = Path(run_config.data).resolve()

    # Generate a container-side lab.toml so `arl eval` etc. work inside.
    # Paths are relative to /workspace (the container's WORKDIR and lab_root).
    rel_backend = f"backend/{Path(lab_config.backend.module).name}"
    container_lab_toml = (
        f'[lab]\nname = "{lab_config.name}"\n'
        f'pipeline_dir = "pipeline"\n'
        f'results_file = "results.tsv"\n'
        f"\n[backend]\n"
        f'module = "{rel_backend}"\n'
        f'class = "{lab_config.backend.cls}"\n'
    )
    # Write to a temp file that gets mounted in
    container_lab_toml_path = verdict_path.parent / "lab.toml"
    container_lab_toml_path.write_text(container_lab_toml)

    cmd = [
        "docker",
        "run",
        "-t",
        "--rm",
        "--add-host=host.docker.internal:host-gateway",
        *auth_env,
    ]

    # Forward host service port if configured
    host_service = lab_config.backend.host_service
    if host_service:
        cmd.extend(
            [
                "-e",
                f"ARL_HOST_SERVICE_URL=http://host.docker.internal:{host_service.port}",
            ]
        )

    cmd.extend(
        [
            # Lab config + backend (read-only)
            "-v",
            f"{container_lab_toml_path}:{CONTAINER_LAB_TOML}:ro",
            "-v",
            f"{backend_dir}:{CONTAINER_BACKEND_DIR}:ro",
            "-v",
            f"{agent_instructions}:{CONTAINER_AGENT_MD}:ro",
            "-v",
            f"{data_dir}:{CONTAINER_DATA_DIR}:ro",
            # Mutable
            "-v",
            f"{pipeline_dir}:{CONTAINER_PIPELINE_DIR}:rw",
            "-v",
            f"{results_file}:{CONTAINER_RESULTS_FILE}:rw",
            "-v",
            f"{verdict_path.parent}:{CONTAINER_VERDICTS_DIR}:rw",
            # Image + Claude Code args
            run_config.docker_image,
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--model",
            run_config.model,
            "--dangerously-skip-permissions",
        ]
    )

    agent_prompt = (
        f"IMPORTANT: Your FIRST action must be to use the Read tool to read "
        f"the file {CONTAINER_AGENT_MD} — it contains your full instructions. "
        f"Do NOT use ToolSearch or any other tool before reading {CONTAINER_AGENT_MD}. "
        f"Do NOT hallucinate or guess the contents of {CONTAINER_AGENT_MD}. "
        f"You MUST make exactly ONE change per experiment, evaluate, then write "
        f"a verdict to {CONTAINER_VERDICT_PATH} and wait for it to be deleted "
        f"before making the next change. Never make multiple changes without "
        f"signaling a verdict between each one."
    )
    if run_config.prompt:
        agent_prompt += "\n\nADDITIONAL INSTRUCTION: " + run_config.prompt

    cmd.append(agent_prompt)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    start_stream_thread(proc.stdout, cost_tracker)
    return proc


def wait_for_verdict(verdict_path: Path, timeout: int) -> dict | None:
    """Wait for verdict.json to appear. Returns parsed verdict or None."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if verdict_path.exists():
            try:
                with open(verdict_path) as f:
                    content = f.read().strip()
                if content:
                    return json.loads(content)
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(1)
    return None


def clear_verdict(verdict_path: Path) -> None:
    """Remove verdict.json so the container knows git ops are done."""
    try:
        verdict_path.unlink()
    except FileNotFoundError:
        pass


def run_session(
    run_config: RunConfig,
    lab_config: LabConfig,
    lab_root: Path,
) -> None:
    """Run the full orchestrated research session."""
    pipeline_dir = (lab_root / lab_config.pipeline_dir).resolve()
    results_tsv = (lab_root / lab_config.results_file).resolve()
    verdict_dir = lab_root / ".verdicts"
    verdict_dir.mkdir(exist_ok=True)
    verdict_path = verdict_dir / VERDICT_FILENAME

    # Ensure results.tsv exists
    results_tsv.touch()

    clear_verdict(verdict_path)

    # Determine starting experiment number from existing results
    existing = read_results(results_tsv)
    start_exp = len(existing)

    state = SessionState(start_time=time.time())

    cost_tracker = CostTracker()

    print("=" * 60)
    print(click.style("Autoresearch Lab — Research Session", bold=True))
    print(f"  Lab:                {lab_config.name}")
    print(f"  Data:               {run_config.data}")
    print(f"  Model:              {run_config.model}")
    print(f"  Max iterations:     {run_config.max_iterations}")
    print(f"  Max hours:          {run_config.max_hours}")
    print(f"  Plateau threshold:  {run_config.plateau_threshold}")
    if run_config.target_score > 0:
        print(f"  Target score:       {run_config.target_score}")
    cost_label = (
        f"${run_config.max_cost:.2f}" if run_config.max_cost > 0 else "unlimited"
    )
    print(f"  Max cost:           {cost_label}")
    if start_exp > 0:
        print(f"  Resuming from:      experiment {start_exp}")
    print("=" * 60)

    # Start host service if configured
    host_service_proc = None
    host_service = lab_config.backend.host_service
    if host_service:
        print(f"\nStarting host service on port {host_service.port}...")
        host_service_proc = start_host_service(
            host_service,
            lab_root,
            data_dir=Path(run_config.data).resolve(),
            pipeline_dir=pipeline_dir,
        )

    def _kill_container(cont: subprocess.Popen) -> None:
        cont.terminate()
        try:
            cont.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cont.kill()

    def _discard_uncommitted(revert_results: bool = False) -> None:
        git_revert(pipeline_dir, lab_root)
        if revert_results:
            git_revert(results_tsv, lab_root)
        for name in (VERDICT_FILENAME, "metrics.json"):
            p = verdict_dir / name
            if p.exists():
                p.unlink()

    def _start_or_restart(
        reason: str,
        after_crash: bool = False,
    ) -> subprocess.Popen:
        _discard_uncommitted(revert_results=after_crash)
        clear_verdict(verdict_path)
        print(f"  {reason}. Starting container...")
        state.last_verdict_time = time.time()
        return start_container(
            run_config, lab_config, lab_root, verdict_path, cost_tracker
        )

    container = _start_or_restart("Starting agent container")

    try:
        while True:
            ret = container.poll()
            if ret is not None:
                state.restarts += 1
                if state.restarts > run_config.max_restarts:
                    print(
                        click.style(
                            f"\nStopping: Container crashed "
                            f"{state.restarts} times "
                            f"(max {run_config.max_restarts})",
                            fg="red",
                        )
                    )
                    break
                container = _start_or_restart(
                    click.style(
                        f"Container exited ({ret}), "
                        f"restart {state.restarts}/"
                        f"{run_config.max_restarts}",
                        fg="yellow",
                    ),
                    after_crash=True,
                )
                continue

            # Check iteration timeout
            if run_config.iteration_timeout > 0:
                elapsed = time.time() - state.last_verdict_time
                if elapsed > run_config.iteration_timeout:
                    _kill_container(container)
                    state.restarts += 1
                    if state.restarts > run_config.max_restarts:
                        print(
                            click.style(
                                f"\nStopping: Iteration timed out "
                                f"{state.restarts} times "
                                f"(max {run_config.max_restarts})",
                                fg="red",
                            )
                        )
                        break
                    container = _start_or_restart(
                        click.style(
                            f"Iteration timed out "
                            f"({run_config.iteration_timeout}s), "
                            f"restart {state.restarts}/"
                            f"{run_config.max_restarts}",
                            fg="yellow",
                        ),
                        after_crash=True,
                    )
                    continue

            # Check stopping conditions
            reason = check_stop_conditions(
                run_config, state, cost_tracker.total_cost_usd
            )
            if reason:
                print(click.style(f"\nStopping: {reason}", fg="cyan"))
                _kill_container(container)
                break

            # Wait for a verdict
            verdict = wait_for_verdict(verdict_path, timeout=30)
            if verdict is None:
                continue

            state.restarts = 0
            state.last_verdict_time = time.time()
            state.iteration += 1

            action = verdict.get("action", "discard")
            exp_num = start_exp + state.iteration
            exp_id = verdict.get("experiment_id", f"exp_{exp_num:03d}")
            notes = verdict.get("notes", "")
            score = float(verdict.get("score", 1.0))
            metrics = verdict.get("metrics", {})

            if action == "keep":
                state.consecutive_discards = 0
                if score < state.best_score:
                    state.best_score = score

                msg = f"{lab_config.name}: {exp_id} — {notes}"
                sha = git_commit(pipeline_dir, msg, lab_root)

                append_result(
                    results_tsv,
                    experiment_id=exp_id,
                    score=score,
                    metrics=metrics,
                    kept=True,
                    commit_sha=sha,
                    notes=notes,
                )
                git_amend_with_results(results_tsv, lab_root)

                metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(
                    f"  [{state.iteration}/"
                    f"{run_config.max_iterations}] "
                    f"{click.style('KEEP', fg='green', bold=True)}  "
                    f"score={score:.4f}  {metrics_str}  {notes}  "
                    f"({sha[:8]})"
                )
            else:
                state.consecutive_discards += 1
                git_revert(pipeline_dir, lab_root)

                append_result(
                    results_tsv,
                    experiment_id=exp_id,
                    score=score,
                    metrics=metrics,
                    kept=False,
                    commit_sha="reverted",
                    notes=notes,
                )

                metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(
                    f"  [{state.iteration}/"
                    f"{run_config.max_iterations}] "
                    f"{click.style('DISCARD', fg='yellow', bold=True)}  "
                    f"score={score:.4f}  {metrics_str}  {notes}  "
                    f"(plateau: "
                    f"{state.consecutive_discards}/"
                    f"{run_config.plateau_threshold})"
                )

            clear_verdict(verdict_path)

    except KeyboardInterrupt:
        print(click.style("\nInterrupted by user", fg="yellow"))
        _kill_container(container)
    finally:
        _discard_uncommitted()
        if host_service_proc:
            host_service_proc.terminate()
            try:
                host_service_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                host_service_proc.kill()

    elapsed = (time.time() - state.start_time) / 60
    print(click.style("\nSession complete:", bold=True))
    print(f"  Iterations: {state.iteration}")
    print(f"  Best score: {state.best_score:.4f}")
    print(f"  Cost:       ${cost_tracker.total_cost_usd:.4f}")
    print(f"  Duration:   {elapsed:.1f} minutes")
