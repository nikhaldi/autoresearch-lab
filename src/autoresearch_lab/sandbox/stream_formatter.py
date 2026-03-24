"""Format Claude Code stream-json output for human-readable terminal display.

Also tracks cumulative API cost from token usage in assistant events.
"""

from __future__ import annotations

import json
import threading
from typing import IO

# Approximate cost per token (USD) by model.
# Prices as of 2026-03. USD per token.
_COST_PER_INPUT_TOKEN = {
    "claude-opus-4-6": 5.0 / 1_000_000,
    "claude-sonnet-4-6": 3.0 / 1_000_000,
    "claude-haiku-4-5": 1.0 / 1_000_000,
}
_COST_PER_OUTPUT_TOKEN = {
    "claude-opus-4-6": 25.0 / 1_000_000,
    "claude-sonnet-4-6": 15.0 / 1_000_000,
    "claude-haiku-4-5": 5.0 / 1_000_000,
}
_COST_PER_CACHE_READ_TOKEN = {
    "claude-opus-4-6": 0.5 / 1_000_000,
    "claude-sonnet-4-6": 0.30 / 1_000_000,
    "claude-haiku-4-5": 0.10 / 1_000_000,
}
_DEFAULT_INPUT_COST = 5.0 / 1_000_000
_DEFAULT_OUTPUT_COST = 25.0 / 1_000_000
_DEFAULT_CACHE_READ_COST = 0.5 / 1_000_000


class CostTracker:
    """Thread-safe accumulator for estimated API cost."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_cost: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return self._total_cost

    def add(self, amount: float) -> None:
        with self._lock:
            self._total_cost += amount

    def add_from_usage(self, usage: dict, model: str) -> None:
        """Estimate cost from a Claude API usage dict."""
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)

        input_cost = _COST_PER_INPUT_TOKEN.get(model, _DEFAULT_INPUT_COST)
        output_cost = _COST_PER_OUTPUT_TOKEN.get(model, _DEFAULT_OUTPUT_COST)
        cache_cost = _COST_PER_CACHE_READ_TOKEN.get(
            model, _DEFAULT_CACHE_READ_COST
        )

        self.add(
            input_tokens * input_cost
            + output_tokens * output_cost
            + cache_read * cache_cost
        )

    def set_from_result(self, cost: float) -> None:
        """Override the estimate with the actual cost from a result event."""
        with self._lock:
            self._total_cost = cost


def start_stream_thread(
    pipe: IO[bytes], cost_tracker: CostTracker
) -> threading.Thread:
    """Spawn a daemon thread that reads stream-json from *pipe* and prints
    formatted output to stdout. Returns the thread (already started)."""
    thread = threading.Thread(
        target=_format_stream, args=(pipe, cost_tracker), daemon=True
    )
    thread.start()
    return thread


def _format_stream(pipe: IO[bytes], tracker: CostTracker) -> None:
    model = ""
    for raw_line in pipe:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(line, flush=True)
            continue
        model = _print_event(event, tracker, model)
    pipe.close()


def _print_event(event: dict, tracker: CostTracker, model: str) -> str:
    """Pretty-print a single Claude Code stream-json event."""
    etype = event.get("type")

    if etype == "system":
        model = event.get("model", model)
        if model:
            print(f"  [system] model={model}", flush=True)

    elif etype == "assistant":
        msg = event.get("message", {})
        usage = msg.get("usage")
        if usage:
            tracker.add_from_usage(usage, model)
        for block in msg.get("content", []):
            btype = block.get("type")
            if btype == "thinking":
                thinking = block.get("thinking", "")
                if thinking:
                    for line in thinking.strip().splitlines():
                        print(f"  [thinking] {line}", flush=True)
            elif btype == "text":
                text = block.get("text", "")
                if text:
                    print(f"  [agent] {text}", flush=True)
            elif btype == "tool_use":
                _print_tool_use(block)

    elif etype == "result":
        cost = event.get("total_cost_usd", 0)
        if cost:
            tracker.set_from_result(cost)
        turns = event.get("num_turns", 0)
        duration = event.get("duration_ms", 0) / 1000
        print(
            f"  [result] {turns} turns, {duration:.1f}s, ${cost:.4f}",
            flush=True,
        )

    return model


def _print_tool_use(block: dict) -> None:
    name = block.get("name", "")
    inp = block.get("input", {})
    if name == "Bash":
        cmd = inp.get("command", "")
        print(f"  [tool] Bash: {cmd[:120]}", flush=True)
    elif name in ("Read", "Glob", "Grep"):
        target = inp.get("file_path", "") or inp.get("pattern", "")
        print(f"  [tool] {name}: {target}", flush=True)
    elif name in ("Edit", "Write"):
        fp = inp.get("file_path", "")
        print(f"  [tool] {name}: {fp}", flush=True)
    else:
        print(f"  [tool] {name}", flush=True)
