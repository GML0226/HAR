from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.data import IMAGE_SUFFIXES

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class FrameSequenceDataset(Dataset):
    """Uniformly samples a single image modality while preserving frame order."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        modality: str,
        image_size: int,
        frames: int,
        labelled: bool = True,
        cache_frame_paths: bool = True,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).fillna("")
        self.modality = modality
        self.image_size = image_size
        self.frames = frames
        self.labelled = labelled
        self.frame_files: list[list[Path]] | None = None
        if cache_frame_paths:
            paths = self.manifest[f"{self.modality}_path"].astype(str).tolist()
            self.frame_files = [self._discover_files(path) for path in tqdm(paths, desc=f"Caching {modality} frame paths", leave=False, dynamic_ncols=True)]

    def __len__(self) -> int:
        return len(self.manifest)

    @staticmethod
    def _discover_files(path: str) -> list[Path]:
        root = Path(path)
        if not path or not root.exists():
            return []
        return sorted(item for item in root.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)

    def _load_clip(self, path: str, cached_files: list[Path] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.zeros((self.frames, 3, self.image_size, self.image_size), dtype=torch.float32)
        files = cached_files if cached_files is not None else self._discover_files(path)
        if not files:
            return output, torch.tensor(0.0)
        indices = np.linspace(0, len(files) - 1, self.frames).round().astype(int)
        loaded = 0
        for target, source in enumerate(indices):
            try:
                with Image.open(files[source]) as image_file:
                    image = image_file.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                    value = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
                output[target] = (value - IMAGENET_MEAN) / IMAGENET_STD
                loaded += 1
            except (OSError, ValueError):
                continue
        return output, torch.tensor(float(loaded > 0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.manifest.iloc[index]
        cached_files = self.frame_files[index] if self.frame_files is not None else None
        frames, valid = self._load_clip(str(row[f"{self.modality}_path"]), cached_files)
        sample: dict[str, torch.Tensor | str] = {"frames": frames, "frame_mask": valid, "clip_id": str(row["clip_id"])}
        if self.labelled:
            sample["label"] = torch.tensor(int(row["action_id"]), dtype=torch.long)
        return sample
