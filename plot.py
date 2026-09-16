"""Plot held-out evaluation success rates from the runs directory."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ALGORITHMS = ["a2c", "ppo", "ddpg", "td3", "sac"]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot evaluation success-rate curves.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--figure-root", default="figures")
    return parser.parse_args()


def read_eval_rows(run_root: Path, mode: str, algorithm: str, seed: int) -> List[Dict[str, float]]:
    path = run_root / mode / algorithm / "seed_{}_eval_metrics.csv".format(seed)
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append({key: float(value) for key, value in row.items() if value not in {"", None}})
    return rows


def aggregate_metric(
    run_root: Path,
    mode: str,
    algorithm: str,
    seeds: Sequence[int],
    metric: str = "success_rate",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Align evaluations by training steps and calculate a 95% normal CI."""

    values_by_step: Dict[int, List[float]] = defaultdict(list)
    available_seeds = 0
    for seed in seeds:
        rows = read_eval_rows(run_root, mode, algorithm, seed)
        if rows:
            available_seeds += 1
        for row in rows:
            if metric in row:
                values_by_step[int(row["total_steps"])].append(row[metric])
    if not values_by_step:
        return np.empty(0), np.empty(0), np.empty(0), available_seeds

    steps = np.asarray(sorted(values_by_step), dtype=np.int64)
    means = np.asarray([np.mean(values_by_step[int(step)]) for step in steps], dtype=np.float64)
    confidence = []
    for step in steps:
        values = np.asarray(values_by_step[int(step)], dtype=np.float64)
        if len(values) < 2:
            confidence.append(0.0)
        else:
            confidence.append(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
    return steps, means, np.asarray(confidence), available_seeds


def plot_mode_learning_curves(
    run_root: Path,
    figure_root: Path,
    mode: str,
    algorithms: Sequence[str],
    seeds: Sequence[int],
) -> Dict[str, float]:
    figure, axis = plt.subplots(figsize=(8, 5))
    final_costs: Dict[str, float] = {}
    plotted = 0
    for algorithm in algorithms:
        steps, means, confidence, available_seeds = aggregate_metric(
            run_root,
            mode,
            algorithm,
            seeds,
        )
        if len(steps) == 0:
            continue
        label = "{} (n={})".format(algorithm.upper(), available_seeds)
        axis.plot(steps, means, label=label)
        axis.fill_between(steps, means - confidence, means + confidence, alpha=0.18)
        final_costs[algorithm] = float(means[-1])
        plotted += 1

    axis.set_title("{}: held-out interception success rate".format(mode.upper()))
    axis.set_xlabel("training environment steps")
    axis.set_ylabel("success rate (higher is better)")
    axis.grid(alpha=0.3)
    if plotted:
        axis.legend()
    figure.tight_layout()
    output_path = figure_root / "{}_learning_curves.png".format(mode)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return final_costs


def plot_best_mode_comparison(
    run_root: Path,
    figure_root: Path,
    best_pn: str,
    best_e2e: str,
    seeds: Sequence[int],
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for mode, algorithm in [("pn", best_pn), ("e2e", best_e2e)]:
        steps, means, confidence, available_seeds = aggregate_metric(
            run_root,
            mode,
            algorithm,
            seeds,
        )
        if len(steps) == 0:
            continue
        label = "{} + {} (n={})".format(mode.upper(), algorithm.upper(), available_seeds)
        axis.plot(steps, means, label=label)
        axis.fill_between(steps, means - confidence, means + confidence, alpha=0.18)
    axis.set_title("Best PN vs best end-to-end method")
    axis.set_xlabel("training environment steps")
    axis.set_ylabel("success rate (higher is better)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_root / "best_pn_vs_best_e2e.png", dpi=180)
    plt.close(figure)


def plot_final_metric_comparison(
    figure_root: Path,
    pn_costs: Dict[str, float],
    e2e_costs: Dict[str, float],
) -> None:
    labels = sorted(set(pn_costs) | set(e2e_costs))
    if not labels:
        return
    x = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(9, 5))
    pn_values = [pn_costs.get(label, np.nan) for label in labels]
    e2e_values = [e2e_costs.get(label, np.nan) for label in labels]
    axis.bar(x - width / 2, pn_values, width, label="PN + RL")
    axis.bar(x + width / 2, e2e_values, width, label="End-to-end RL")
    axis.set_xticks(x, [label.upper() for label in labels])
    axis.set_ylabel("final success rate (higher is better)")
    axis.set_title("Final held-out metric comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(figure_root / "final_metric_comparison.png", dpi=180)
    plt.close(figure)


def main():
    arguments = parse_arguments()
    run_root = Path(arguments.run_root)
    figure_root = Path(arguments.figure_root)
    figure_root.mkdir(parents=True, exist_ok=True)
    pn_costs = plot_mode_learning_curves(
        run_root,
        figure_root,
        "pn",
        arguments.algorithms,
        arguments.seeds,
    )
    e2e_costs = plot_mode_learning_curves(
        run_root,
        figure_root,
        "e2e",
        arguments.algorithms,
        arguments.seeds,
    )
    if pn_costs and e2e_costs:
        best_pn = max(pn_costs, key=pn_costs.get)
        best_e2e = max(e2e_costs, key=e2e_costs.get)
        plot_best_mode_comparison(run_root, figure_root, best_pn, best_e2e, arguments.seeds)
    plot_final_metric_comparison(figure_root, pn_costs, e2e_costs)
    print("saved figures in {}".format(figure_root))


if __name__ == "__main__":
    main()
