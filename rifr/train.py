"""Stage A: HR teacher; Stage B: student on HR+LR with KL + shared Softmax."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rifr.data.celeba import CelebAIdentityDataset, downsample_to_scale, resolve_celeba_root
from rifr.models.student_teacher import FaceNet


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def kl_div_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(p, q, reduction="batchmean") * (T * T)


def train(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_celeba_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    image_size = int(cfg["model"].get("image_size", 112))
    train_ds = CelebAIdentityDataset(
        root,
        split="train",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=seed,
        train=True,
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    teacher = FaceNet(
        train_ds.num_classes,
        embed_dim=int(cfg["model"].get("embed_dim", 512)),
        pretrained=bool(cfg["model"].get("pretrained_backbone", True)),
    ).to(device)
    student = FaceNet(
        train_ds.num_classes,
        embed_dim=int(cfg["model"].get("embed_dim", 512)),
        pretrained=bool(cfg["model"].get("pretrained_backbone", True)),
    ).to(device)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    lr = float(cfg["train"].get("lr", 0.1))
    mom = float(cfg["train"].get("momentum", 0.9))
    wd = float(cfg["train"].get("weight_decay", 5e-4))
    history: List[dict] = []
    t0 = time.time()

    # ---- Stage A: HR teacher ----
    opt_t = torch.optim.SGD(teacher.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    te = int(cfg["train"].get("teacher_epochs", 8))
    teacher.train()
    for epoch in range(1, te + 1):
        losses = []
        pbar = tqdm(loader, desc=f"teacher {epoch}/{te}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(teacher(images), labels)
            opt_t.zero_grad(set_to_none=True)
            loss.backward()
            opt_t.step()
            losses.append(float(loss.detach()))
            pbar.set_postfix(loss=f"{losses[-1]:.3f}")
        mean_l = float(np.mean(losses))
        history.append({"stage": "teacher", "epoch": epoch, "total": mean_l})
        print(f"[teacher] epoch={epoch} loss={mean_l:.4f}")

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # Share Softmax weights (paper)
    student.share_classifier_from(teacher)
    # train backbone+fc only (classifier shared / tied)
    student_params = [p for n, p in student.named_parameters() if "classifier" not in n]
    opt_s = torch.optim.SGD(student_params, lr=lr, momentum=mom, weight_decay=wd)

    scales = [float(s) for s in cfg["model"].get("lr_scales", [1.0, 0.5, 0.25])]
    T = float(cfg["train"].get("kl_temp", 4.0))
    w_kl = float(cfg["train"].get("kl_weight", 1.0))
    w_ce = float(cfg["train"].get("ce_weight", 1.0))
    se = int(cfg["train"].get("student_epochs", 12))

    student.train()
    for epoch in range(1, se + 1):
        losses = []
        pbar = tqdm(loader, desc=f"student {epoch}/{se}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            # randomly pick a scale per batch (includes HR=1.0)
            scale = float(scales[int(torch.randint(0, len(scales), (1,)).item())])
            x = downsample_to_scale(images, scale)
            with torch.no_grad():
                # teacher always sees HR
                t_logits = teacher(images)
            s_logits = student(x)
            ce = F.cross_entropy(s_logits, labels)
            kl = kl_div_logits(s_logits, t_logits, T)
            loss = w_ce * ce + w_kl * kl
            opt_s.zero_grad(set_to_none=True)
            loss.backward()
            opt_s.step()
            losses.append({"total": float(loss.detach()), "ce": float(ce.detach()), "kl": float(kl.detach())})
            pbar.set_postfix(loss=f"{losses[-1]['total']:.3f}")
        mean = {k: float(np.mean([d[k] for d in losses])) for k in losses[0]}
        mean.update({"stage": "student", "epoch": epoch})
        history.append(mean)
        print(f"[student] epoch={epoch} total={mean['total']:.4f} ce={mean['ce']:.4f} kl={mean['kl']:.4f}")

    elapsed = time.time() - t0
    ckpt = {
        "teacher": teacher.state_dict(),
        "student": student.state_dict(),
        "num_classes": train_ds.num_classes,
        "config": cfg,
    }
    torch.save(ckpt, ckpt_dir / "last.pt")
    summary = {
        "protocol": cfg["train"].get("protocol", "full"),
        "teacher_epochs": te,
        "student_epochs": se,
        "n_train": len(train_ds),
        "num_classes": train_ds.num_classes,
        "lr_scales": scales,
        "elapsed_sec": elapsed,
        "history": history,
        "checkpoint": str(ckpt_dir / "last.pt"),
    }
    with open(log_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {log_dir / 'train_summary.json'} elapsed={elapsed:.1f}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
