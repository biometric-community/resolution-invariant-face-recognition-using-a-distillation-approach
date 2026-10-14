"""Figures + REPORT.md from real metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rifr.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(history, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for stage, color in [("teacher", MULTI_SERIES_COLORS[0]), ("student", MULTI_SERIES_COLORS[1])]:
        hs = [h for h in history if h.get("stage") == stage]
        if hs:
            ax.plot(
                [h["epoch"] for h in hs],
                [h["total"] for h in hs],
                color=color,
                linewidth=LINEWIDTH_MAIN,
                label=stage,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Teacher / student training loss")
    ax.legend()
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_scales(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    scales = []
    t_acc, s_acc = [], []
    for k, v in sorted(eval_res.get("scales", {}).items(), key=lambda kv: float(kv[0].split("_")[1])):
        scales.append(k.replace("scale_", ""))
        t_acc.append(v["teacher"]["accuracy"] * 100)
        s_acc.append(v["student"]["accuracy"] * 100)
    x = np.arange(len(scales))
    w = 0.35
    ax.bar(x - w / 2, t_acc, w, color=MULTI_SERIES_COLORS[0], label="Teacher (HR)", edgecolor="#002080")
    ax.bar(x + w / 2, s_acc, w, color=MULTI_SERIES_COLORS[1], label="Student (HR+LR)", edgecolor="#800000")
    ax.set_xticks(x)
    ax.set_xticklabels([f"×{s}" for s in scales])
    ax.set_ylabel("LFW accuracy (%)")
    ax.set_title("Verification across resolutions")
    ax.legend()
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    lines = [
        f"# REPORT: Resolution-Invariant Distillation — {protocol}",
        "",
        f"**Protocol:** `{protocol}`  ",
        f"**Package:** `rifr`  ",
        "",
        "## Summary (LFW)",
        "",
        "| Scale | Teacher Acc | Teacher EER | Student Acc | Student EER |",
        "|-------|-------------|-------------|-------------|-------------|",
    ]
    for k, v in sorted(eval_res.get("scales", {}).items(), key=lambda kv: -float(kv[0].split("_")[1])):
        t, s = v["teacher"], v["student"]
        lines.append(
            f"| {k} | {t['accuracy']*100:.2f}% | {t['eer']*100:.2f}% | "
            f"{s['accuracy']*100:.2f}% | {s['eer']*100:.2f}% |"
        )
    lines += [
        "",
        "## Training",
        "",
        f"- Teacher epochs: {train_s.get('teacher_epochs')}",
        f"- Student epochs: {train_s.get('student_epochs')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- LR scales: {train_s.get('lr_scales')}",
        f"- Elapsed: {train_s.get('elapsed_sec', 0):.2f} s",
        f"- Checkpoint: `{train_s.get('checkpoint')}`",
        "",
        "## Figures",
        "",
        "### `train_loss.svg`",
        "",
        "![train_loss](figures/train_loss.svg)",
        "",
        "### `lfw_by_scale.svg`",
        "",
        "![lfw_by_scale](figures/lfw_by_scale.svg)",
        "",
        "## Reproduce",
        "",
        "```bash",
        "bash scripts/smoke.sh",
        "bash scripts/train_full.sh",
        "```",
        "",
        "## Notes",
        "",
        "- CelebA substitutes MegaFace/FaceScrub for HR teacher pretraining (DEVIATIONS.md).",
        "- Synthetic LR via downsample→upsample; TinyFace/SCFace not available.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval_results.json").read_text())
    plot_train(train_s.get("history", []), fig_dir / "train_loss.svg")
    plot_scales(eval_res, fig_dir / "lfw_by_scale.svg")
    tracked = _ROOT / "figures"
    tracked.mkdir(exist_ok=True)
    for name in ("train_loss.svg", "lfw_by_scale.svg"):
        src = fig_dir / name
        if src.exists():
            (tracked / name).write_bytes(src.read_bytes())
    write_report(cfg, train_s, eval_res, _ROOT / "REPORT.md")
    print(f"Wrote {_ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
