#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plot the public EuroSAT StatefulDataLoader worker-start benchmark.

Usage from the TorchData repository root:

    python benchmarks/stateful_dataloader/plot_worker_start_benchmark.py \
        --results benchmarks/stateful_dataloader/eurosat_spawn_results.json \
        --output /tmp/eurosat_worker_start_benchmark.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure


SERIAL_COLOR = "#2E86AB"
PARALLEL_COLOR = "#F18F01"
GRID_COLOR = "#D8DEE4"
TEXT_COLOR = "#202124"


def _load_results(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as results_file:
        return json.load(results_file)


def _values(
    configuration: dict[str, Any],
    mode: str,
    metric: str,
) -> list[float]:
    return [float(trial[metric]) for trial in configuration[mode]]


def _summary(values: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(np.median(values)),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 90)),
    )


def _series(
    configurations: Sequence[dict[str, Any]],
    mode: str,
    metric: str,
) -> tuple[list[float], list[list[float]]]:
    summaries = [_summary(_values(config, mode, metric)) for config in configurations]
    medians = [summary[0] for summary in summaries]
    errors = [
        [median - low for median, low, _ in summaries],
        [high - median for median, _, high in summaries],
    ]
    return medians, errors


def _style_axis(ax: Axes) -> None:
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=TEXT_COLOR)


def _add_raw_points(
    ax: Axes,
    configurations: Sequence[dict[str, Any]],
    mode: str,
    metric: str,
    positions: np.ndarray,
    color: str,
) -> None:
    for position, configuration in zip(positions, configurations):
        values = _values(configuration, mode, metric)
        offsets = np.linspace(-0.045, 0.045, len(values))
        ax.scatter(
            position + offsets,
            values,
            s=20,
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            zorder=4,
        )


def _label_bars(
    ax: Axes,
    bars: BarContainer,
    values: Sequence[float],
    unit: str,
) -> None:
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:,.1f}{unit}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=TEXT_COLOR,
        )


def _comparison_labels(
    ax: Axes,
    serial: Sequence[float],
    parallel: Sequence[float],
    lower_is_better: bool,
) -> None:
    top = ax.get_ylim()[1]
    for index, (serial_value, parallel_value) in enumerate(zip(serial, parallel)):
        if serial_value == 0:
            ax.text(
                index,
                top * 0.075,
                "n/a",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=TEXT_COLOR,
            )
            continue
        change = (parallel_value / serial_value - 1) * 100
        if lower_is_better:
            label = f"{abs(change):.1f}% lower" if change < 0 else f"{change:.1f}% higher"
        else:
            label = f"{change:+.1f}%"
        ax.text(
            index,
            max(serial_value, parallel_value) + top * 0.075,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=TEXT_COLOR,
        )


def _grouped_metric(
    ax: Axes,
    configurations: Sequence[dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
    unit: str,
    lower_is_better: bool,
) -> None:
    serial, serial_errors = _series(configurations, "serial", metric)
    parallel, parallel_errors = _series(configurations, "parallel", metric)
    x = np.arange(len(configurations))
    width = 0.36
    serial_positions = x - width / 2
    parallel_positions = x + width / 2
    serial_bars = ax.bar(
        serial_positions,
        serial,
        width,
        yerr=serial_errors,
        capsize=3,
        color=SERIAL_COLOR,
        alpha=0.82,
        label="Serial spawn (p=1)",
    )
    parallel_bars = ax.bar(
        parallel_positions,
        parallel,
        width,
        yerr=parallel_errors,
        capsize=3,
        color=PARALLEL_COLOR,
        alpha=0.82,
        label="Parallel spawn (p=num_workers)",
    )
    _add_raw_points(ax, configurations, "serial", metric, serial_positions, SERIAL_COLOR)
    _add_raw_points(
        ax,
        configurations,
        "parallel",
        metric,
        parallel_positions,
        PARALLEL_COLOR,
    )
    ax.set_ylim(0, max(*serial, *parallel) * 1.28)
    _label_bars(ax, serial_bars, serial, unit)
    _label_bars(ax, parallel_bars, parallel, unit)
    _comparison_labels(ax, serial, parallel, lower_is_better)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, [config["label"] for config in configurations])
    _style_axis(ax)


def _paired_startup_panel(ax: Axes, configuration: dict[str, Any]) -> None:
    serial = _values(configuration, "serial", "time_to_first_batch_s")
    parallel = _values(configuration, "parallel", "time_to_first_batch_s")
    for serial_value, parallel_value in zip(serial, parallel):
        ax.plot([0, 1], [serial_value, parallel_value], color="#9AA0A6", alpha=0.7)
        ax.scatter(0, serial_value, color=SERIAL_COLOR, s=42, zorder=3)
        ax.scatter(1, parallel_value, color=PARALLEL_COLOR, s=42, zorder=3)
    serial_median = float(np.median(serial))
    parallel_median = float(np.median(parallel))
    speedups = np.array(serial) / np.array(parallel)
    speedup_summary = f"Paired median {np.median(speedups):.2f}x faster\n"
    speedup_range = f"range {np.min(speedups):.2f}-{np.max(speedups):.2f}x"
    ax.hlines(serial_median, -0.18, 0.18, color=TEXT_COLOR, linewidth=3)
    ax.hlines(parallel_median, 0.82, 1.18, color=TEXT_COLOR, linewidth=3)
    ax.text(
        0.5,
        0.53,
        speedup_summary + speedup_range,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": GRID_COLOR},
    )
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0, max(serial) * 1.13)
    parallelism = configuration["parallelism"]
    worker_count = configuration["num_workers"]
    ax.set_xticks(
        [0, 1],
        ["Serial spawn\n(p=1)", f"Parallel spawn\n(p={parallelism})"],
    )
    ax.set_ylabel("Time to first batch (seconds)")
    ax.set_title(
        f"B. Primary {worker_count}-worker result: every matched trial improves",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    _style_axis(ax)


def _primary_configuration(
    configurations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    for configuration in configurations:
        if configuration["num_workers"] == 8 and configuration["dataset_copies"] == 1:
            return configuration
    raise ValueError("Results must include the primary 8-worker EuroSAT configuration")


def _median_metric(configuration: dict[str, Any], mode: str, metric: str) -> float:
    return _summary(_values(configuration, mode, metric))[0]


def _build_figure(results: dict[str, Any]) -> Figure:
    configurations = results["configurations"]
    primary = _primary_configuration(configurations)
    startup_metric = "time_to_first_batch_s"
    throughput_metric = "steady_state_samples_per_s"
    serial_primary = _median_metric(primary, "serial", startup_metric)
    parallel_primary = _median_metric(primary, "parallel", startup_metric)
    startup_reduction = (1 - parallel_primary / serial_primary) * 100
    throughput_serial = _median_metric(primary, "serial", throughput_metric)
    throughput_parallel = _median_metric(primary, "parallel", throughput_metric)
    throughput_change = (throughput_parallel / throughput_serial - 1) * 100
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    _grouped_metric(
        axes[0, 0],
        configurations,
        "time_to_first_batch_s",
        "A. Startup latency falls as worker count grows",
        "Time to first batch (seconds; lower is better)",
        "s",
        True,
    )
    _paired_startup_panel(axes[0, 1], primary)
    _grouped_metric(
        axes[1, 0],
        configurations,
        "steady_state_samples_per_s",
        "C. Steady-state throughput remains comparable",
        "Samples per second (higher is better)",
        "",
        False,
    )
    _grouped_metric(
        axes[1, 1],
        configurations,
        "memory_increase_mb",
        "D. Process-tree PSS increases modestly",
        "Peak PSS increase (MiB; lower is better)",
        "",
        True,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        "StatefulDataLoader concurrent spawn: public EuroSAT benchmark",
        fontsize=19,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.985,
    )
    fig.text(
        0.5,
        0.946,
        f"Primary 8-worker result: {serial_primary:.2f} s -> "
        f"{parallel_primary:.2f} s time-to-first-batch "
        f"(-{startup_reduction:.1f}%, {serial_primary / parallel_primary:.2f}x ratio "
        f"of medians) while steady-state throughput changes {throughput_change:+.1f}%",
        ha="center",
        fontsize=12,
        color=TEXT_COLOR,
    )
    benchmark = results["benchmark"]
    fig.text(
        0.5,
        0.012,
        f"{benchmark['dataset']} ({benchmark['dataset_size']:,} JPEGs) | "
        f"batch {benchmark['batch_size']} | spawn | "
        f"{benchmark['cpu']} CPU-only host | fresh process per configuration | "
        "primary trials alternated AB/BA | bars=median, whiskers=p10-p90, circles=raw trials | "
        "all primary first-batch SHA-256 digests matched",
        ha="center",
        fontsize=9,
        color="#5F6368",
    )
    fig.subplots_adjust(
        top=0.86,
        bottom=0.10,
        left=0.075,
        right=0.98,
        hspace=0.36,
        wspace=0.24,
    )
    return fig


def _parse_args() -> argparse.Namespace:
    description = "Plot the public EuroSAT concurrent-spawn benchmark"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = _load_results(args.results)
    figure = _build_figure(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
