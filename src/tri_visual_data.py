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

VISUAL_MODALITIES = ("Depth_Color", "IR", "Thermal")
NORMALIZATION = {
    "Depth_Color": (torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)),
    "IR": (torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1), torch.tensor([0.25, 0.25, 0.25]).view(3, 1, 1)),
    "Thermal": (torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1), torch.tensor([0.25, 0.25, 0.25]).view(3, 1, 1)),
}


class TriVisualDataset(Dataset):
    """Clip-level aligned Depth_Color, IR, and Thermal input with missing-modality masks."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_size: int,
        frames: int,
        training: bool,
        labelled: bool = True,
        cache_frame_paths: bool = True,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).fillna("")
        self.image_size = image_size
        self.frames = frames
        self.training = training
        self.labelled = labelled
        self._validate_manifest()
        self.file_cache: dict[str, list[list[Path]]] | None = self._cache_files() if cache_frame_paths else None

    def _validate_manifest(self) -> None:
        required = {"clip_id", *[f"{modality}_path" for modality in VISUAL_MODALITIES]}
        if self.labelled:
            required |= {"action_id", "user_id", "trial_id"}
        missing = sorted(required - set(self.manifest.columns))
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")
        if self.manifest.clip_id.duplicated().any():
            raise ValueError("Manifest contains duplicate clip_id values")
        if self.labelled:
            trial_keys = self.manifest[["action_id", "user_id", "trial_id"]]
            if trial_keys.duplicated().any():
                raise ValueError("Manifest contains duplicate action/user/trial keys")
            expected = self.manifest.apply(lambda row: f"{int(row.action_id)}_{row.user_id}_{row.trial_id}", axis=1)
            if not (expected == self.manifest.clip_id).all():
                raise ValueError("clip_id does not match action_id/user_id/trial_id")

    @staticmethod
    def _discover_files(path: str) -> list[Path]:
        root = Path(path)
        if not path or not root.exists():
            return []
        return sorted(item for item in root.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)

    def _cache_files(self) -> dict[str, list[list[Path]]]:
        cache: dict[str, list[list[Path]]] = {}
        for modality in VISUAL_MODALITIES:
            paths = self.manifest[f"{modality}_path"].astype(str).tolist()
            cache[modality] = [
                self._discover_files(path)
                for path in tqdm(paths, desc=f"Caching {modality} frame paths", leave=False, dynamic_ncols=True)
            ]
        return cache

    def __len__(self) -> int:
        return len(self.manifest)

    def _sample_indices(self, length: int) -> np.ndarray:
        boundaries = np.linspace(0, length, self.frames + 1).astype(int)
        indices: list[int] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            end = max(end, start + 1)
            if self.training:
                indices.append(random.randrange(start, end))
            else:
                indices.append((start + end - 1) // 2)
        return np.clip(np.asarray(indices), 0, length - 1)

    def _augment(self, image: Image.Image, modality: str) -> Image.Image:
        if not self.training:
            return image
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if modality in {"IR", "Thermal"}:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        return image

    def _load_modality(self, files: list[Path], modality: str) -> tuple[torch.Tensor, float]:
        output = torch.zeros((self.frames, 3, self.image_size, self.image_size), dtype=torch.float32)
        if not files:
            return output, 0.0
        mean, std = NORMALIZATION[modality]
        loaded = 0
        for target, source in enumerate(self._sample_indices(len(files))):
            try:
                with Image.open(files[source]) as image_file:
                    image = image_file.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                    image = self._augment(image, modality)
                    value = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
                output[target] = (value - mean) / std
                loaded += 1
            except (OSError, ValueError):
                continue
        return output, loaded / self.frames

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.manifest.iloc[index]
        frames, quality = [], []
        for modality in VISUAL_MODALITIES:
            files = self.file_cache[modality][index] if self.file_cache is not None else self._discover_files(str(row[f"{modality}_path"]))
            value, valid_fraction = self._load_modality(files, modality)
            frames.append(value)
            quality.append(valid_fraction)
        quality_tensor = torch.tensor(quality, dtype=torch.float32)
        sample: dict[str, torch.Tensor | str] = {
            "frames": torch.stack(frames),
            "modality_mask": quality_tensor.gt(0),
            "quality": quality_tensor,
            "clip_id": str(row.clip_id),
            "user_id": str(row.user_id) if self.labelled else "",
        }
        if self.labelled:
            sample["label"] = torch.tensor(int(row.action_id), dtype=torch.long)
        return sample
