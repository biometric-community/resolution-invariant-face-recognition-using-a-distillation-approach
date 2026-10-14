"""Embed one image with student (optionally downsampled)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rifr.data.celeba import downsample_to_scale
from rifr.models.student_teacher import FaceNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--which", choices=["student", "teacher"], default="student")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
        Path(cfg["paths"]["checkpoint_dir"]) / "last.pt",
        map_location=device,
        weights_only=False,
    )
    n = int(ckpt["num_classes"])
    teacher = FaceNet(n, pretrained=False).to(device)
    student = FaceNet(n, pretrained=False).to(device)
    teacher.load_state_dict(ckpt["teacher"])
    student.load_state_dict(ckpt["student"])
    student.share_classifier_from(teacher)
    model = student if args.which == "student" else teacher
    model.eval()
    tf = transforms.Compose(
        [
            transforms.Resize((int(cfg["model"].get("image_size", 112)),) * 2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = tf(Image.open(args.image).convert("RGB"))
    x = downsample_to_scale(x, args.scale).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.embed(x).cpu().numpy()[0].tolist()
    out = {"which": args.which, "scale": args.scale, "embedding": emb}
    if args.out:
        Path(args.out).write_text(json.dumps(out) + "\n")
    print(json.dumps({"which": args.which, "scale": args.scale, "dim": len(emb)}))


if __name__ == "__main__":
    main()
