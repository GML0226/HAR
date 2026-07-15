from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

IMAGE_MODALITIES = ("Depth_Color", "IR", "Thermal")
ALL_MODALITIES = (*IMAGE_MODALITIES, "Skeleton", "IMU", "Radar")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _resample(sequence: np.ndarray, length: int) -> np.ndarray:
    if len(sequence) == 0:
        return np.zeros((length, sequence.shape[1]), dtype=np.float32)
    index = np.linspace(0, len(sequence) - 1, length).round().astype(np.int64)
    return sequence[index].astype(np.float32, copy=False)


def _normalise_channels(sequence: np.ndarray) -> np.ndarray:
    if not np.isfinite(sequence).any():
        return np.zeros_like(sequence, dtype=np.float32)
    sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
    mean = sequence.mean(axis=0, keepdims=True)
    std = sequence.std(axis=0, keepdims=True).clip(min=1e-5)
    return ((sequence - mean) / std).astype(np.float32)


def _numeric_csv(path: Path, width: int) -> np.ndarray:
    try:
        table = pd.read_csv(path, encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError):
        table = pd.read_csv(path, encoding="gb18030")
    numeric = table.apply(pd.to_numeric, errors="coerce")
    keep = [column for column in numeric if numeric[column].notna().mean() >= 0.5]
    values = numeric[keep].to_numpy(dtype=np.float32) if keep else np.empty((0, 0), dtype=np.float32)
    if values.shape[1] < width:
        values = np.pad(values, ((0, 0), (0, width - values.shape[1])))
    return values[:, :width]


class MultimodalDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        image_size: int,
        image_frames: int,
        sequence_length: int,
        labelled: bool = True,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).fillna("")
        self.image_size = image_size
        self.image_frames = image_frames
        self.sequence_length = sequence_length
        self.labelled = labelled
        self.file_cache = self._cache_files()

    def __len__(self) -> int:
        return len(self.manifest)

    @staticmethod
    def _files(path: str, suffixes: set[str]) -> list[Path]:
        if not path:
            return []
        root = Path(path)
        if not root.exists():
            return []
        return sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in suffixes)

    def _cache_files(self) -> dict[str, list[list[Path]]]:
        suffixes = {
            "Depth_Color": IMAGE_SUFFIXES,
            "IR": IMAGE_SUFFIXES,
            "Thermal": IMAGE_SUFFIXES,
            "Skeleton": {".json"},
            "IMU": {".csv"},
            "Radar": {".csv"},
        }
        cache: dict[str, list[list[Path]]] = {}
        for modality in ALL_MODALITIES:
            paths = self.manifest[f"{modality}_path"].astype(str).tolist()
            cache[modality] = [
                self._files(path, suffixes[modality])
                for path in tqdm(paths, desc=f"Caching {modality} files", leave=False, dynamic_ncols=True)
            ]
        return cache

    def _load_images(self, files: list[Path]) -> tuple[torch.Tensor, float]:
        output = torch.zeros((self.image_frames, 3, self.image_size, self.image_size), dtype=torch.float32)
        if not files:
            return output, 0.0
        indices = np.linspace(0, len(files) - 1, self.image_frames).round().astype(int)
        loaded = 0
        for destination, index in enumerate(indices):
            try:
                with Image.open(files[index]) as image_file:
                    image = image_file.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
                output[destination] = torch.from_numpy(array)
                loaded += 1
            except (OSError, ValueError):
                continue
        return output, float(loaded > 0)

    def _load_skeleton(self, files: list[Path]) -> tuple[torch.Tensor, float]:
        frames: list[np.ndarray] = []
        for file in files:
            try:
                payload = json.loads(file.read_text(encoding="utf-8"))
                people = payload if isinstance(payload, list) else []
                keypoints = people[0].get("keypoints", []) if people else []
                points = np.asarray(keypoints, dtype=np.float32)
                if points.shape != (17, 3):
                    continue
                points -= points.mean(axis=0, keepdims=True)
                scale = np.linalg.norm(points, axis=-1).mean()
                frames.append(points / max(float(scale), 1e-4))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        if not frames:
            return torch.zeros((self.sequence_length, 102), dtype=torch.float32), 0.0
        sequence = _resample(np.stack(frames).reshape(len(frames), -1), self.sequence_length)
        velocity = np.diff(sequence, axis=0, prepend=sequence[:1])
        return torch.from_numpy(np.concatenate([sequence, velocity], axis=1)), 1.0

    def _load_imu(self, files: list[Path]) -> tuple[torch.Tensor, float]:
        chunks: list[np.ndarray] = []
        for file in files[:2]:
            try:
                chunks.append(_resample(_normalise_channels(_numeric_csv(file, 6)), self.sequence_length))
            except (OSError, ValueError, pd.errors.EmptyDataError):
                continue
        if not chunks:
            return torch.zeros((self.sequence_length, 12), dtype=torch.float32), 0.0
        while len(chunks) < 2:
            chunks.append(np.zeros_like(chunks[0]))
        return torch.from_numpy(np.concatenate(chunks, axis=1)), 1.0

    def _load_radar(self, files: list[Path]) -> tuple[torch.Tensor, float]:
        if not files:
            return torch.zeros((self.sequence_length, 16), dtype=torch.float32), 0.0
        try:
            table = pd.read_csv(files[0])
            expected = ["x", "y", "z", "v", "snr"]
            if not set(expected).issubset(table.columns):
                raise ValueError("Unexpected radar columns")
            grouped = table.groupby("frame", sort=True)[expected]
            mean = grouped.mean().to_numpy(np.float32)
            std = grouped.std().fillna(0.0).to_numpy(np.float32)
            count = grouped.size().to_numpy(np.float32)[:, None]
            maximum = grouped.max().to_numpy(np.float32)
            values = np.concatenate([mean, std, maximum, count], axis=1)
            values = _resample(_normalise_channels(values), self.sequence_length)
            return torch.from_numpy(values), 1.0
        except (OSError, ValueError, KeyError, pd.errors.EmptyDataError):
            return torch.zeros((self.sequence_length, 16), dtype=torch.float32), 0.0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.manifest.iloc[index]
        image_values, image_masks = [], []
        for modality in IMAGE_MODALITIES:
            value, mask = self._load_images(self.file_cache[modality][index])
            image_values.append(value)
            image_masks.append(mask)
        skeleton, skeleton_mask = self._load_skeleton(self.file_cache["Skeleton"][index])
        imu, imu_mask = self._load_imu(self.file_cache["IMU"][index])
        radar, radar_mask = self._load_radar(self.file_cache["Radar"][index])
        sample: dict[str, torch.Tensor | str] = {
            "images": torch.stack(image_values),
            "skeleton": skeleton,
            "imu": imu,
            "radar": radar,
            "modality_mask": torch.tensor(image_masks + [skeleton_mask, imu_mask, radar_mask], dtype=torch.float32),
            "clip_id": str(row["clip_id"]),
        }
        if self.labelled:
            sample["label"] = torch.tensor(int(row["action_id"]), dtype=torch.long)
        return sample


def split_by_users(manifest: pd.DataFrame, validation_users: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = manifest[manifest.user_id.isin(validation_users)].copy()
    train = manifest[~manifest.user_id.isin(validation_users)].copy()
    overlap = set(train.user_id) & set(validation.user_id)
    if overlap or train.empty or validation.empty:
        raise ValueError(f"Invalid subject split. Overlap: {sorted(overlap)}")
    return train, validation
