"""LFW pairs with optional resolution scale."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from rifr.data.celeba import downsample_to_scale


def resolve_lfw_root(cfg: dict) -> Path:
    root = Path(cfg["data"].get("lfw_root", "../../datasets/lfw"))
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[2] / root).resolve()
    if not (root / "lfw").is_dir() or not (root / "pairs.txt").is_file():
        raise FileNotFoundError(root)
    return root


def _parse_pairs(pairs_path: Path, max_pairs: Optional[int] = None):
    out = []
    for line in pairs_path.read_text().strip().splitlines()[1:]:
        toks = line.split()
        if len(toks) == 3:
            name, n1, n2 = toks
            a = f"{name}/{name}_{int(n1):04d}.jpg"
            b = f"{name}/{name}_{int(n2):04d}.jpg"
            out.append((a, b, 1))
        elif len(toks) == 4:
            n1, i1, n2, i2 = toks
            a = f"{n1}/{n1}_{int(i1):04d}.jpg"
            b = f"{n2}/{n2}_{int(i2):04d}.jpg"
            out.append((a, b, 0))
        if max_pairs is not None and len(out) >= max_pairs:
            break
    return out


class LFWPairsDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_size: int = 112,
        max_pairs: Optional[int] = None,
        scale: float = 1.0,
    ):
        self.img_dir = Path(root) / "lfw"
        self.scale = float(scale)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.pairs = _parse_pairs(Path(root) / "pairs.txt", max_pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        a, b, label = self.pairs[index]
        img_a = self.transform(Image.open(self.img_dir / a).convert("RGB"))
        img_b = self.transform(Image.open(self.img_dir / b).convert("RGB"))
        if self.scale < 1.0:
            img_a = downsample_to_scale(img_a, self.scale)
            img_b = downsample_to_scale(img_b, self.scale)
        return img_a, img_b, label
