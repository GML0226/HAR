from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import MultimodalDataset, split_by_users
from src.model import SixModalHAR
from src.utils import ensure_dir, load_config, save_json, seed_everything


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, epoch: int, total_epochs: int) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} validation", leave=False, dynamic_ncols=True)
        for batch in progress:
            batch = move_batch(batch, device)
            output = model(batch)["logits"].argmax(dim=1)
            labels.extend(batch["label"].cpu().tolist())
            predictions.extend(output.cpu().tolist())
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "samples": len(labels),
    }
    return metrics, np.asarray(labels), np.asarray(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    output_dir = ensure_dir(config["output_dir"])
    shutil.copy2(args.config, output_dir / "config.yaml")

    manifest = pd.read_csv(config["data"]["train_manifest"]).fillna("")
    train_frame, validation_frame = split_by_users(manifest, config["split"]["validation_users"])
    train_users, validation_users = set(train_frame.user_id), set(validation_frame.user_id)
    if train_users & validation_users:
        raise RuntimeError("Subject leakage detected")
    for section, limit in (("train", config["training"].get("max_train_samples")), ("validation", config["training"].get("max_validation_samples"))):
        if limit:
            frame = train_frame if section == "train" else validation_frame
            reduced = frame.sample(min(int(limit), len(frame)), random_state=config["seed"]).sort_index()
            if section == "train":
                train_frame = reduced
            else:
                validation_frame = reduced
    split_report = {
        "train_users": sorted(train_users),
        "validation_users": sorted(validation_users),
        "train_samples": len(train_frame),
        "validation_samples": len(validation_frame),
        "overlap": sorted(train_users & validation_users),
    }
    save_json(split_report, output_dir / "split_report.json")
    print(split_report)

    common = config["data"]
    train_data = MultimodalDataset(train_frame, common["image_size"], common["image_frames"], common["sequence_length"])
    validation_data = MultimodalDataset(validation_frame, common["image_size"], common["image_frames"], common["sequence_length"])
    loader_args = {"batch_size": config["training"]["batch_size"], "num_workers": config["training"]["num_workers"], "pin_memory": torch.cuda.is_available()}
    if config["training"]["num_workers"] > 0:
        loader_args.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = SixModalHAR(**config["model"], num_classes=common["num_classes"]).to(device, memory_format=torch.channels_last)
    print(f"Device={device}; GPU={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'none'}; parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["training"]["amp"] and device.type == "cuda"))
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy, stale_epochs, history = -1.0, 0, []
    for epoch in range(1, config["training"]["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        print(f"Epoch {epoch}/{config['training']['epochs']} started; lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']} training", leave=False, dynamic_ncols=True)
        for batch_index, batch in enumerate(progress, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(batch)["logits"]
                loss = loss_fn(logits, batch["label"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(train_loader):
                progress.set_postfix(loss=f"{losses[-1]:.4f}", avg_loss=f"{np.mean(losses):.4f}")
        scheduler.step()
        metrics, labels, predictions = evaluate(model, validation_loader, device, epoch, config["training"]["epochs"])
        metrics.update({"epoch": epoch, "train_loss": float(np.mean(losses)), "learning_rate": optimizer.param_groups[0]["lr"], "epoch_seconds": round(time.perf_counter() - epoch_start, 1)})
        history.append(metrics)
        print(f"Epoch {epoch}/{config['training']['epochs']} finished in {metrics['epoch_seconds']:.1f}s; train_loss={metrics['train_loss']:.4f}; val_accuracy={metrics['accuracy']:.4f}; macro_f1={metrics['macro_f1']:.4f}; best_accuracy={best_accuracy:.4f}", flush=True)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy, stale_epochs = metrics["accuracy"], 0
            torch.save({"model_state": model.state_dict(), "config": config, "metrics": metrics}, output_dir / "best.pt")
            pd.DataFrame({"clip_id": validation_frame.clip_id, "label": labels, "prediction": predictions}).to_csv(output_dir / "validation_predictions.csv", index=False)
        else:
            stale_epochs += 1
            if stale_epochs >= config["training"]["early_stopping_patience"]:
                break
    checkpoint = output_dir / "best.pt"
    profile = {
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_megabytes": round(checkpoint.stat().st_size / 1024 / 1024, 3),
        "under_100_mb": checkpoint.stat().st_size <= 100 * 1024 * 1024,
        "device": str(device),
    }
    save_json({"history": history, "best_accuracy": best_accuracy}, output_dir / "metrics.json")
    save_json(profile, output_dir / "model_profile.json")
    print(profile)


if __name__ == "__main__":
    main()
