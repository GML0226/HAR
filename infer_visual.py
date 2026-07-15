from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils import load_config
from src.visual_data import FrameSequenceDataset
from src.visual_model import ResNet18MeanPool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet18_depth.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/resnet18_depth/submission.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    manifest = pd.read_csv(data["test_manifest"]).fillna("")
    dataset = FrameSequenceDataset(manifest, data["modality"], data["image_size"], data["frames"], labelled=False)
    loader_args = {"batch_size": config["training"]["batch_size"], "shuffle": False, "num_workers": config["training"]["num_workers"], "pin_memory": torch.cuda.is_available()}
    if config["training"]["num_workers"] > 0:
        loader_args.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(dataset, **loader_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = ResNet18MeanPool(num_classes=data["num_classes"], **config["model"]).to(device, memory_format=torch.channels_last)
    model.load_state_dict(saved["model_state"])
    model.eval()
    print(f"Device={device}; samples={len(dataset)}; batch_size={config['training']['batch_size']}", flush=True)
    predictions: list[int] = []
    with torch.no_grad():
        progress = tqdm(loader, desc="Test inference", dynamic_ncols=True)
        for batch in progress:
            logits = model(batch["frames"].to(device), batch["frame_mask"].to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    submission = pd.DataFrame({"path": manifest.submission_path, "prediction": predictions})
    if len(submission) != 405 or submission.path.duplicated().any() or not submission.prediction.between(0, 39).all():
        raise RuntimeError("Submission validation failed")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
    print(f"Wrote {len(submission)} validated predictions to {destination}")


if __name__ == "__main__":
    main()
