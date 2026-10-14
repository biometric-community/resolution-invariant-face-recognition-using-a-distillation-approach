"""LFW verification at multiple resolutions for teacher vs student."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rifr.data.lfw import LFWPairsDataset, resolve_lfw_root
from rifr.models.student_teacher import FaceNet


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc, best_eer = 0.0, 1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        best_acc = max(best_acc, float((pred == labels).mean()))
        pos, neg = labels == 1, labels == 0
        if pos.any() and neg.any():
            fpr = float((pred[neg] == 1).mean())
            fnr = float((pred[pos] == 0).mean())
            best_eer = min(best_eer, 0.5 * (fpr + fnr))
    return best_acc, best_eer


@torch.no_grad()
def eval_model(model: FaceNet, loader, device) -> Dict:
    model.eval()
    scores, labels = [], []
    for a, b, y in tqdm(loader, leave=False):
        ea = model.embed(a.to(device))
        eb = model.embed(b.to(device))
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    acc, eer = best_acc_eer(s, y)
    return {"accuracy": acc, "eer": eer, "n_pairs": int(len(y))}


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    lfw = resolve_lfw_root(cfg)
    image_size = int(cfg["model"].get("image_size", 112))
    max_pairs = cfg["eval"].get("max_lfw_pairs")
    max_pairs = int(max_pairs) if max_pairs is not None else None
    scales = [float(s) for s in cfg["eval"].get("eval_scales", [1.0, 0.5, 0.25])]

    ckpt = torch.load(
        Path(cfg["paths"]["checkpoint_dir"]) / "last.pt",
        map_location=device,
        weights_only=False,
    )
    n_classes = int(ckpt["num_classes"])
    teacher = FaceNet(n_classes, pretrained=False).to(device)
    student = FaceNet(n_classes, pretrained=False).to(device)
    teacher.load_state_dict(ckpt["teacher"])
    student.load_state_dict(ckpt["student"])
    # re-tie shared classifier for student after load
    student.share_classifier_from(teacher)

    results = {"scales": {}}
    for scale in scales:
        ds = LFWPairsDataset(lfw, image_size=image_size, max_pairs=max_pairs, scale=scale)
        loader = DataLoader(
            ds,
            batch_size=int(cfg["eval"].get("batch_size", 128)),
            shuffle=False,
            num_workers=int(cfg["train"].get("num_workers", 4)),
        )
        t_m = eval_model(teacher, loader, device)
        s_m = eval_model(student, loader, device)
        key = f"scale_{scale}"
        results["scales"][key] = {"teacher": t_m, "student": s_m}
        print(
            f"[{key}] teacher acc={t_m['accuracy']:.4f} eer={t_m['eer']:.4f} | "
            f"student acc={s_m['accuracy']:.4f} eer={s_m['eer']:.4f}"
        )

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
