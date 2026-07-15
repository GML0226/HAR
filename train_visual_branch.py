from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import split_by_users
from src.tri_visual_model import VisualBranch
from src.utils import ensure_dir, load_config, save_json, seed_everything
from src.visual_data import FrameSequenceDataset


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    labels, logits, clip_ids, user_ids = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False, dynamic_ncols=True):
            clip_ids.extend(batch["clip_id"])
            user_ids.extend(batch["user_id"])
            batch = move_batch(batch, device)
            _, output = model(batch["frames"], batch["frame_mask"])
            labels.extend(batch["label"].cpu().tolist())
            logits.append(output.cpu().numpy())
    logits_array = np.concatenate(logits)
    labels_array = np.asarray(labels)
    predictions = logits_array.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "macro_f1": float(f1_score(labels_array, predictions, average="macro", zero_division=0)),
        "samples": len(labels_array),
    }, labels_array, logits_array, clip_ids, user_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    output_dir = ensure_dir(config["output_dir"])
    shutil.copy2(args.config, output_dir / "config.yaml")
    manifest = pd.read_csv(config["data"]["train_manifest"]).fillna("")
    train_frame, validation_frame = split_by_users(manifest, config["split"]["validation_users"])
    if set(train_frame.user_id) & set(validation_frame.user_id):
        raise RuntimeError("Subject leakage detected")
    data = config["data"]
    train_set = FrameSequenceDataset(train_frame, data["modality"], data["image_size"], data["frames"], training=True)
    validation_set = FrameSequenceDataset(validation_frame, data["modality"], data["image_size"], data["frames"], training=False)
    loader_args = {"batch_size": config["training"]["batch_size"], "num_workers": config["training"]["num_workers"], "pin_memory": torch.cuda.is_available()}
    if loader_args["num_workers"] > 0:
        loader_args.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = VisualBranch(data["backbone"], config["model"]["pretrained"], data["num_classes"]).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["training"]["amp"] and device.type == "cuda"))
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy, stale, history = -1.0, 0, []
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.set_trainable("frozen" if epoch <= config["training"]["freeze_backbone_epochs"] else "full")
        model.train()
        start = time.perf_counter()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']} training", leave=False, dynamic_ncols=True):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                _, logits = model(batch["frames"], batch["frame_mask"])
                loss = loss_fn(logits, batch["label"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics, labels, logits, clip_ids, user_ids = evaluate(model, validation_loader, device)
        metrics.update({"epoch": epoch, "train_loss": float(np.mean(losses)), "seconds": round(time.perf_counter() - start, 1)})
        history.append(metrics)
        print(metrics, flush=True)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy, stale = metrics["accuracy"], 0
            torch.save({"model_state": model.state_dict(), "config": config, "metrics": metrics}, output_dir / "best.pt")
            np.savez_compressed(
                output_dir / "validation_logits.npz",
                logits=logits,
                labels=labels,
                clip_ids=np.asarray(clip_ids),
                user_ids=np.asarray(user_ids),
                prediction_source=np.asarray(config["prediction_source"]),
            )
        else:
            stale += 1
            if stale >= config["training"]["early_stopping_patience"]:
                break
    checkpoint = output_dir / "best.pt"
    save_json({"history": history, "best_accuracy": best_accuracy}, output_dir / "metrics.json")
    save_json({"parameters": int(sum(parameter.numel() for parameter in model.parameters())), "checkpoint_megabytes": round(checkpoint.stat().st_size / 1024 / 1024, 3), "under_100_mb": checkpoint.stat().st_size <= 100, "prediction_source": config["prediction_source"]}, output_dir / "model_profile.json")


if __name__ == "__main__":
    main()
