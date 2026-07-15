from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.data import IMAGE_SUFFIXES

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SENSOR_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
SENSOR_STD = torch.tensor([0.25, 0.25, 0.25]).view(3, 1, 1)


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
        training: bool = False,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).fillna("")
        self.modality = modality
        self.image_size = image_size
        self.frames = frames
        self.labelled = labelled
        self.training = training
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

    def _sample_indices(self, length: int) -> np.ndarray:
        boundaries = np.linspace(0, length, self.frames + 1).astype(int)
        indices: list[int] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            end = max(end, start + 1)
            indices.append(random.randrange(start, end) if self.training else (start + end - 1) // 2)
        return np.clip(np.asarray(indices), 0, length - 1)

    def _load_clip(self, path: str, cached_files: list[Path] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.zeros((self.frames, 3, self.image_size, self.image_size), dtype=torch.float32)
        files = cached_files if cached_files is not None else self._discover_files(path)
        if not files:
            return output, torch.tensor(0.0)
        indices = self._sample_indices(len(files))
        mean, std = (IMAGENET_MEAN, IMAGENET_STD) if self.modality == "Depth_Color" else (SENSOR_MEAN, SENSOR_STD)
        loaded = 0
        for target, source in enumerate(indices):
            try:
                with Image.open(files[source]) as image_file:
                    image = image_file.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                    if self.training and random.random() < 0.5:
                        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    if self.training and self.modality in {"IR", "Thermal"}:
                        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
                        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
                    value = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
                output[target] = (value - mean) / std
                loaded += 1
            except (OSError, ValueError):
                continue
        return output, torch.tensor(float(loaded > 0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.manifest.iloc[index]
        cached_files = self.frame_files[index] if self.frame_files is not None else None
        frames, valid = self._load_clip(str(row[f"{self.modality}_path"]), cached_files)
        sample: dict[str, torch.Tensor | str] = {
            "frames": frames,
            "frame_mask": valid,
            "clip_id": str(row["clip_id"]),
            "user_id": str(row["user_id"]) if self.labelled and "user_id" in row else "",
        }
        if self.labelled:
            sample["label"] = torch.tensor(int(row["action_id"]), dtype=torch.long)
        return sample
