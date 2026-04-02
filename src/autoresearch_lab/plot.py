"""Experiment progress plotting.

This module requires matplotlib, which is an optional dependency.
Install with: pip install 'autoresearch-lab[plot]'
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def _plot_series(ax, rows, label, color, show_labels=True):
    """Plot a single results series on the given axes."""
    experiments = []
    for i, row in enumerate(rows):
        experiments.append(
            {
                "number": i + 1,
                "score": float(row.get("score", "0")),
                "kept": row.get("kept") == "yes",
                "notes": row.get("notes", ""),
            }
        )

    numbers = [e["number"] for e in experiments]

    kept_numbers = [e["number"] for e in experiments if e["kept"]]
    kept_scores = [e["score"] for e in experiments if e["kept"]]
    kept_notes = [e["notes"] for e in experiments if e["kept"]]

    disc_numbers = [e["number"] for e in experiments if not e["kept"]]
    disc_scores = [e["score"] for e in experiments if not e["kept"]]

    # Compute running best (step function through kept experiments)
    running_best_x = []
    running_best_y = []
    best_so_far = float("inf")
    for e in experiments:
        if e["kept"] and e["score"] < best_so_far:
            best_so_far = e["score"]
            running_best_x.append(e["number"])
            running_best_y.append(best_so_far)
    # Extend to the last experiment so the line reaches the end
    if running_best_x and numbers:
        running_best_x.append(numbers[-1])
        running_best_y.append(running_best_y[-1])

    light_color = to_rgba(color, alpha=0.08)

    ax.scatter(
        disc_numbers,
        disc_scores,
        c=[light_color],
        s=20,
        alpha=0.4,
        zorder=2,
    )
    ax.scatter(
        kept_numbers,
        kept_scores,
        c=color,
        s=60,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )
    ax.step(
        running_best_x,
        running_best_y,
        where="post",
        color=color,
        alpha=0.7,
        linewidth=2,
        label=label,
        zorder=3,
    )

    for x, y, note in zip(kept_numbers, kept_scores, kept_notes):
        if show_labels and note:
            text = f"{note} ({y:.2f})" if len(note) < 40 else f"{note[:37]}..."
            ax.annotate(
                text,
                (x, y),
                fontsize=6,
                rotation=20,
                ha="left",
                va="bottom",
                alpha=0.8,
            )

    n_kept = len(kept_numbers)
    n_total = len(experiments)

    # Update the step-line legend entry with counts
    ax.lines[-1].set_label(f"{label} ({n_total} experiments, {n_kept} kept)")


def plot_results(
    series: list[tuple[str, list[dict[str, str]]]],
    output: str | None = None,
    *,
    title: str | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    show_labels: bool = True,
) -> None:
    """Build and display/save an experiment progress chart.

    Args:
        series: List of (label, rows) tuples, where rows come
            from ``read_results()``.
        output: File path to save the plot. Shows interactively
            if *None*.
        title: Custom chart title.
        ymin: Optional lower y-axis limit.
        ymax: Optional upper y-axis limit to exclude outliers.
        ylabel: Custom y-axis label.
        xlabel: Custom x-axis label.
        figsize: Figure size as (width, height) in inches.
        show_labels: Whether to annotate kept experiments.
    """
    fig, ax = plt.subplots(figsize=figsize or (14, 7))

    # Default matplotlib color cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    for i, (label, s_rows) in enumerate(series):
        color = colors[i % len(colors)]
        _plot_series(ax, s_rows, label, color, show_labels=show_labels)

    ax.set_xlim(left=0)
    ax.margins(y=0.02)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_title(title or "Autoresearch Lab Progress")
    ax.set_xlabel(xlabel or "Experiment #")
    ax.set_ylabel(ylabel or "Score (lower is better)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()
