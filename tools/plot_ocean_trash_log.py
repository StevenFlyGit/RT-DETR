import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def load_runs(path: str) -> List[List[Dict[str, Any]]]:
    runs: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    last_epoch = None

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" not in data:
                continue

            epoch = data.get("epoch")
            if last_epoch is not None and epoch is not None and epoch < last_epoch:
                if current:
                    runs.append(current)
                current = []

            current.append(data)
            last_epoch = epoch

    if current:
        runs.append(current)

    return runs


def best_record(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    for item in records:
        metrics = item.get("test_coco_eval_bbox")
        if not metrics:
            continue
        ap = metrics[0]
        if best is None or ap > best["ap"]:
            best = {"ap": ap, "epoch": item.get("epoch"), "metrics": metrics}
    return best or {"ap": None, "epoch": None, "metrics": None}


def plot_runs(runs: List[List[Dict[str, Any]]], output_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_loss, ax_ap = axes

    for idx, run in enumerate(runs, start=1):
        epochs: List[int] = []
        losses: List[float] = []
        aps: List[float] = []

        for item in run:
            epoch = item.get("epoch")
            loss = item.get("train_loss")
            metrics = item.get("test_coco_eval_bbox")
            ap = metrics[0] if metrics else None

            if epoch is None or loss is None or ap is None:
                continue

            epochs.append(epoch)
            losses.append(loss)
            aps.append(ap)

        label = f"run {idx}"
        ax_loss.plot(epochs, losses, label=label)
        ax_ap.plot(epochs, aps, label=label)

    ax_loss.set_title("Train Loss")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linestyle="--", alpha=0.4)

    ax_ap.set_title("Val AP (COCO bbox AP)")
    ax_ap.set_xlabel("epoch")
    ax_ap.set_ylabel("AP")
    ax_ap.grid(True, linestyle="--", alpha=0.4)

    ax_loss.legend(loc="best")
    ax_ap.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RT-DETR training log curves.")
    parser.add_argument(
        "--log",
        default="e:/StevenWork/RT-DETR/output/ocean_trash_rtdetr/log.txt",
        help="Path to log.txt",
    )
    parser.add_argument(
        "--out",
        default="e:/StevenWork/RT-DETR/output/ocean_trash_rtdetr/plots/train_curves.png",
        help="Output image path",
    )
    args = parser.parse_args()

    runs = load_runs(args.log)
    if not runs:
        raise SystemExit("No runs found in log.")

    all_records = [item for run in runs for item in run]
    best = best_record(all_records)
    print("best_ap", best["ap"])
    print("epoch", best["epoch"])
    print("test_coco_eval_bbox", best["metrics"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_runs(runs, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
